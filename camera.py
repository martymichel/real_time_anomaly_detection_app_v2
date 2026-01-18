# Copyright (C) 2025, IDS Imaging Development Systems GmbH.
# Modified version with best practices and bug fixes.

from typing import cast, Iterator
from contextlib import contextmanager
import logging

from ids_peak_afl import ids_peak_afl
from ids_peak_common import Range, PixelFormat, CommonException, Channel
from ids_peak import ids_peak, ImageView
from ids_peak.ids_peak import (
    Device,
    NodeMap,
    DataStream,
    Buffer,
    Timeout,
    DataStreamFlushMode_DiscardAll,
    AcquisitionStopMode_Default,
)

logger = logging.getLogger(__name__)


class Camera:
    """IDS Peak Camera wrapper with proper resource management."""
    
    # Empfohlene Buffer-Anzahl für stabile Acquisition
    RECOMMENDED_BUFFER_COUNT = 5
    
    def __init__(
        self,
        userset: str = "UserSet0",
        apply_optimizations: bool = False,
        max_framerate: bool = True,
        reverse_x: bool = False,  # Horizontal spiegeln
        reverse_y: bool = True,   # Vertikal spiegeln (für Kivy/OpenGL)
    ) -> None:
        """
        Initialisiert Kamera und lädt definiertes UserSet.

        Args:
            userset: UserSet zum Laden ("UserSet0", "UserSet1", "Default", etc.)
            apply_optimizations: Wenn True, werden nach dem Laden Optimierungen
                                 angewendet (RGB8, Auto-Features aus).
                                 Wenn False, bleiben UserSet-Einstellungen unverändert.
            max_framerate: Wenn True, wird Framerate auf Maximum gesetzt.
            reverse_x: Wenn True, wird Bild horizontal gespiegelt
            reverse_y: Wenn True, wird Bild vertikal gespiegelt (Kivy/OpenGL)
        """
        self._device: Device | None = None
        self._data_stream: DataStream | None = None
        self._acquisition_running = False
        self._node_cache: dict[str, ids_peak.Node] = {}
        self._reverse_x = reverse_x
        self._reverse_y = reverse_y

        # Library-Initialisierung (nur wenn noch nicht initialisiert)
        try:
            if not ids_peak.Library.IsInitialized():
                ids_peak.Library.Initialize()
                logger.debug("IDS Peak Library initialisiert")
        except (AttributeError, RuntimeError):
            # Fallback falls IsInitialized() nicht existiert
            try:
                ids_peak.Library.Initialize()
                logger.debug("IDS Peak Library initialisiert (Fallback)")
            except Exception:
                pass  # Schon initialisiert

        try:
            if not ids_peak_afl.Library.IsInitialized():
                ids_peak_afl.Library.Init()
                logger.debug("AFL Library initialisiert")
        except (AttributeError, RuntimeError):
            # Fallback falls IsInitialized() nicht existiert
            try:
                ids_peak_afl.Library.Init()
                logger.debug("AFL Library initialisiert (Fallback)")
            except Exception:
                pass  # Schon initialisiert
        
        try:
            self._open_device()
            self._setup_data_stream()
            
            # Definierten Startzustand laden
            self._load_userset(userset)
            
            # Koordinatensystem für Kivy/OpenGL (ReverseY)
            self._fix_coordinates()
            
            if apply_optimizations:
                # Host-seitige Auto-Features nutzen (überschreibt UserSet!)
                self._disable_device_autofeatures()
                
                # Pixel-Format optimieren (überschreibt UserSet!)
                self._optimize_pixel_format()
            
            # Maximale Framerate (unabhängig von apply_optimizations)
            if max_framerate:
                self._set_max_framerate()
            
            # Reconnect-Handler registrieren
            self._setup_reconnect_handler()
            
            # Geladene Einstellungen ausgeben
            self._log_current_settings()
            
        except Exception:
            self.close()
            raise
    
    def _set_max_framerate(self) -> None:
        """Setzt Framerate auf Maximum."""
        try:
            max_fps = self.framerate_range.maximum
            self.framerate = max_fps
            logger.info(f"Framerate auf Maximum gesetzt: {max_fps:.1f} fps")
        except CommonException as e:
            logger.warning(f"Framerate konnte nicht gesetzt werden: {e}")
    
    def _log_current_settings(self) -> None:
        """Gibt aktuelle Kamera-Einstellungen aus."""
        try:
            settings = {
                "Pixel Format": str(self.pixel_format),
                "Resolution": f"{self._get_width()}x{self._get_height()}",
                "Exposure": f"{self.exposure:.1f} µs",
                "Framerate": f"{self.framerate:.2f} fps",
                "Master Gain": f"{self.master_gain:.2f}",
            }

            # RGB Gains falls verfügbar
            try:
                self._set_gain_selector("Red")
                settings["Red Gain"] = f"{self.red_gain:.2f}"
                settings["Green Gain"] = f"{self.green_gain:.2f}"
                settings["Blue Gain"] = f"{self.blue_gain:.2f}"
            except CommonException:
                pass

            # Image orientation
            settings["ReverseX (Horizontal)"] = "ON" if self.reverse_x else "OFF"
            settings["ReverseY (Vertical)"] = "ON" if self.reverse_y else "OFF"

            logger.info("=== Aktuelle Kamera-Einstellungen (aus UserSet) ===")
            for key, value in settings.items():
                logger.info(f"  {key}: {value}")
            logger.info("=" * 50)

        except CommonException as e:
            logger.warning(f"Einstellungen konnten nicht ausgelesen werden: {e}")
    
    def _get_width(self) -> int:
        """Gibt aktuelle Bildbreite zurück."""
        return cast(ids_peak.IntegerNode, self._get_node("Width")).Value()
    
    def _get_height(self) -> int:
        """Gibt aktuelle Bildhöhe zurück."""
        return cast(ids_peak.IntegerNode, self._get_node("Height")).Value()
    
    def _open_device(self) -> None:
        """Öffnet erstes verfügbares Gerät."""
        instance = ids_peak.DeviceManager.Instance()
        instance.Update()
        
        devices = [
            dev for dev in instance.Devices()
            if dev.IsOpenable(ids_peak.DeviceAccessType_Control)
        ]
        
        if not devices:
            raise RuntimeError("Keine Kamera verfügbar")
        
        self._device = devices[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        self._remote_node_map = self._device.RemoteDevice().NodeMaps()[0]
        logger.info(f"Kamera geöffnet: {devices[0].DisplayName()}")
    
    def _setup_data_stream(self) -> None:
        """Initialisiert DataStream."""
        if self._device is None:
            raise RuntimeError("Device nicht initialisiert")
        self._data_stream = self._device.DataStreams()[0].OpenDataStream()
    
    def _get_node(self, name: str) -> ids_peak.Node:
        """Cached Node-Lookup."""
        if name not in self._node_cache:
            self._node_cache[name] = self.remote_device_nodemap.FindNode(name)
        return self._node_cache[name]
    
    def _try_get_node(self, name: str) -> ids_peak.Node | None:
        """Cached Node-Lookup mit None bei Fehler."""
        if name not in self._node_cache:
            node = self.remote_device_nodemap.TryFindNode(name)
            if node is not None:
                self._node_cache[name] = node
            return node
        return self._node_cache[name]
    
    def _load_userset(self, userset: str = "UserSet0") -> None:
        """
        Lädt UserSet aus Kamera-Speicher.
        
        Fallback auf "Default" wenn gewünschtes UserSet nicht verfügbar.
        """
        try:
            selector = cast(ids_peak.EnumerationNode, self._get_node("UserSetSelector"))
            available = [e.SymbolicValue() for e in selector.AvailableEntries()]
            
            target = userset if userset in available else "Default"
            if target != userset:
                logger.warning(f"UserSet '{userset}' nicht verfügbar, nutze '{target}'")
            
            selector.SetCurrentEntry(target)
            cast(ids_peak.CommandNode, self._get_node("UserSetLoad")).Execute()
            logger.info(f"UserSet '{target}' geladen")
            
        except CommonException as e:
            logger.error(f"UserSet-Load fehlgeschlagen: {e}")
    
    def _fix_coordinates(self) -> None:
        """Spiegelt Bild horizontal und/oder vertikal für Anwendungs-Koordinatensystem."""
        # Horizontal spiegeln (ReverseX)
        node_x = cast(ids_peak.BooleanNode | None, self._try_get_node("ReverseX"))
        if node_x is not None and node_x.IsAvailable():
            try:
                if not node_x.IsWriteable():
                    was_started = self._acquisition_running
                    if was_started:
                        self.stop_acquisition()
                    node_x.SetValue(self._reverse_x)
                    if was_started:
                        self.start_acquisition()
                else:
                    node_x.SetValue(self._reverse_x)
                logger.debug(f"ReverseX gesetzt: {self._reverse_x}")
            except CommonException as e:
                logger.warning(f"ReverseX konnte nicht gesetzt werden: {e}")

        # Vertikal spiegeln (ReverseY) - für Kivy/OpenGL
        node_y = cast(ids_peak.BooleanNode | None, self._try_get_node("ReverseY"))
        if node_y is not None and node_y.IsAvailable():
            try:
                if not node_y.IsWriteable():
                    was_started = self._acquisition_running
                    if was_started:
                        self.stop_acquisition()
                    node_y.SetValue(self._reverse_y)
                    if was_started:
                        self.start_acquisition()
                else:
                    node_y.SetValue(self._reverse_y)
                logger.debug(f"ReverseY gesetzt: {self._reverse_y}")
            except CommonException as e:
                logger.warning(f"ReverseY konnte nicht gesetzt werden: {e}")
    
    def _disable_device_autofeatures(self) -> None:
        """Deaktiviert kameraseitige Auto-Features für Host-Kontrolle."""
        auto_nodes = ["ExposureAuto", "BalanceWhiteAuto", "GainAuto", "FocusAuto"]
        
        for node_name in auto_nodes:
            node = cast(ids_peak.EnumerationNode | None, self._try_get_node(node_name))
            if node is None:
                continue
            try:
                node.SetCurrentEntry("Off")
                logger.debug(f"{node_name} deaktiviert")
            except CommonException as e:
                logger.debug(f"{node_name} konnte nicht deaktiviert werden: {e}")
    
    def _optimize_pixel_format(self) -> None:
        """Setzt optimales Pixel-Format (RGB8/BGR8 bevorzugt)."""
        # Packed Formate ausschließen
        if self.pixel_format in [PixelFormat.RGB_10_PACKED_32, PixelFormat.BGR_10_PACKED_32]:
            try:
                bayer_formats = [f for f in self.pixel_format_list if f.has_channel(Channel.BAYER)]
                if bayer_formats:
                    self.pixel_format = bayer_formats[0]
            except CommonException as e:
                logger.warning(f"Packed-Format-Fallback fehlgeschlagen: {e}")
        
        # RGB8/BGR8 bevorzugen für Performance
        try:
            available = self.pixel_format_list
            if PixelFormat.RGB_8 in available:
                self.pixel_format = PixelFormat.RGB_8
                logger.info("Pixel-Format auf RGB8 optimiert")
            elif PixelFormat.BGR_8 in available:
                self.pixel_format = PixelFormat.BGR_8
                logger.info("Pixel-Format auf BGR8 optimiert")
        except CommonException as e:
            logger.warning(f"Pixel-Format-Optimierung fehlgeschlagen: {e}")
    
    def _setup_reconnect_handler(self) -> None:
        """Registriert Reconnect-Callback."""
        try:
            system_node_map = self.device.ParentInterface().ParentSystem().NodeMaps()[0]
            reconnect_node = cast(
                ids_peak.BooleanNode | None,
                system_node_map.TryFindNode("ReconnectEnable")
            )
            if reconnect_node is not None:
                reconnect_node.SetValue(True)
                logger.info("Reconnect aktiviert")
        except CommonException as e:
            logger.debug(f"Reconnect nicht verfügbar: {e}")
        
        instance = ids_peak.DeviceManager.Instance()
        self._reconnect_callback = instance.DeviceReconnectedCallback(self._on_device_reconnected)
        self._reconnect_handle = instance.RegisterDeviceReconnectedCallback(self._reconnect_callback)
    
    def _on_device_reconnected(
        self,
        device: ids_peak.Device,
        info: ids_peak.DeviceReconnectInformation,
    ) -> None:
        """Handler für Gerät-Reconnect."""
        logger.info("Gerät reconnected")
        
        # Node-Cache invalidieren
        self._node_cache.clear()
        
        if info.IsSuccessful():
            self._fix_coordinates()
            self._disable_device_autofeatures()
            return
        
        # Payload-Size prüfen
        payload_size = cast(ids_peak.IntegerNode, self._get_node("PayloadSize")).Value()
        announced = self.data_stream.AnnouncedBuffers()
        
        if announced and payload_size != announced[0].Size():
            logger.info("Payload-Size geändert, Buffer neu allozieren")
            self.stop_acquisition()
            self._fix_coordinates()
            self._disable_device_autofeatures()
            self.start_acquisition()
        elif not info.IsRemoteDeviceAcquisitionRunning():
            cast(ids_peak.CommandNode, self._get_node("AcquisitionStart")).Execute()
    
    def close(self) -> None:
        """Sauberes Herunterfahren aller Ressourcen."""
        if self._acquisition_running:
            try:
                self.stop_acquisition()
            except Exception as e:
                logger.error(f"Fehler beim Stoppen der Acquisition: {e}")
        
        if self._data_stream is not None:
            try:
                self._data_stream = None
            except Exception as e:
                logger.error(f"Fehler beim Schließen des DataStream: {e}")
        
        if self._device is not None:
            try:
                self._device = None
            except Exception as e:
                logger.error(f"Fehler beim Schließen des Device: {e}")
        
        self._node_cache.clear()
        
        try:
            ids_peak_afl.Library.Exit()
            ids_peak.Library.Close()
        except Exception as e:
            logger.error(f"Fehler beim Library-Cleanup: {e}")
        
        logger.info("Kamera geschlossen")
    
    def __enter__(self) -> "Camera":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
    
    def __del__(self) -> None:
        # Fallback falls close() nicht aufgerufen wurde
        if self._device is not None:
            logger.warning("Camera wurde nicht explizit geschlossen, nutze close() oder Context Manager")
            self.close()
    
    def start_acquisition(self) -> None:
        """Startet Bilderfassung mit ausreichend Buffern."""
        if self._acquisition_running:
            return
        
        payload_size = cast(ids_peak.IntegerNode, self._get_node("PayloadSize")).Value()
        
        # Mehr Buffer für Stabilität
        buffer_count = max(
            self.data_stream.NumBuffersAnnouncedMinRequired(),
            self.RECOMMENDED_BUFFER_COUNT
        )
        
        for _ in range(buffer_count):
            buffer = self.data_stream.AllocAndAnnounceBuffer(payload_size)
            self.data_stream.QueueBuffer(buffer)
        
        cast(ids_peak.IntegerNode, self._get_node("TLParamsLocked")).SetValue(1)
        
        self._acquisition_running = True
        
        self.data_stream.StartAcquisition()
        acq_start = cast(ids_peak.CommandNode, self._get_node("AcquisitionStart"))
        acq_start.Execute()
        acq_start.WaitUntilDone()
        
        logger.info(f"Acquisition gestartet ({buffer_count} Buffer)")
    
    def stop_acquisition(self) -> None:
        """Stoppt Bilderfassung und gibt Buffer frei."""
        if not self._acquisition_running:
            return
        
        self._acquisition_running = False
        
        acq_stop = cast(ids_peak.CommandNode, self._get_node("AcquisitionStop"))
        acq_stop.Execute()
        acq_stop.WaitUntilDone()
        
        if self.data_stream.IsGrabbing():
            self.data_stream.StopAcquisition(AcquisitionStopMode_Default)
        self.data_stream.Flush(DataStreamFlushMode_DiscardAll)
        
        for buffer in self.data_stream.AnnouncedBuffers():
            self.data_stream.RevokeBuffer(buffer)
        
        cast(ids_peak.IntegerNode, self._get_node("TLParamsLocked")).SetValue(0)
        
        logger.info("Acquisition gestoppt")
    
    @contextmanager
    def capture_frame(self, timeout_ms: int = 5000) -> Iterator[Buffer]:
        """
        Context Manager für sicheren Frame-Zugriff.
        
        Gibt Buffer zurück (hat Width(), Height(), PixelFormat(), BasePtr()).
        Für ImageView: buffer.ToImageView()
        
        Buffer wird automatisch zurückgegeben.
        
        Usage:
            with camera.capture_frame() as buffer:
                w, h = buffer.Width(), buffer.Height()
                view = buffer.ToImageView()  # falls nötig
            # Buffer automatisch zurück in Queue
        """
        buffer = self.data_stream.WaitForFinishedBuffer(Timeout(timeout_ms))
        try:
            yield buffer
        finally:
            self.data_stream.QueueBuffer(buffer)
    
    def wait_for_image_view(self, timeout_ms: int = 5000) -> tuple[ImageView, Buffer]:
        """
        Wartet auf nächstes Bild.
        
        WICHTIG: Caller muss queue_buffer(buffer) aufrufen!
        Besser: capture_frame() Context Manager nutzen.
        
        Returns:
            Tuple aus ImageView und Buffer (für queue_buffer und Width/Height)
        """
        buffer = self.data_stream.WaitForFinishedBuffer(Timeout(timeout_ms))
        return buffer.ToImageView(), buffer
    
    def queue_buffer(self, buffer: Buffer) -> None:
        """Gibt Buffer zurück in Queue."""
        self.data_stream.QueueBuffer(buffer)
    
    def kill_datastream_wait(self) -> None:
        """Bricht wartenden WaitForFinishedBuffer ab."""
        self.data_stream.KillWait()
    
    # Properties
    
    @property
    def device(self) -> Device:
        if self._device is None:
            raise RuntimeError("Device nicht initialisiert")
        return self._device
    
    @property
    def remote_device_nodemap(self) -> NodeMap:
        return self._remote_node_map
    
    @property
    def data_stream(self) -> DataStream:
        if self._data_stream is None:
            raise RuntimeError("DataStream nicht initialisiert")
        return self._data_stream
    
    @property
    def acquisition_running(self) -> bool:
        return self._acquisition_running
    
    def _range_from_node(self, node_name: str) -> Range:
        node = cast(
            ids_peak.IntegerNode | ids_peak.FloatNode,
            self._get_node(node_name),
        )
        increment_type = node.IncrementType()
        node_type = node.Type()
        
        if increment_type == ids_peak.NodeIncrementType_NoIncrement:
            default_inc = 0.0 if node_type is ids_peak.NodeType_Float else 0
            return Range(node.Minimum(), node.Maximum(), default_inc)
        elif increment_type == ids_peak.NodeIncrementType_FixedIncrement:
            return Range(node.Minimum(), node.Maximum(), node.Increment())
        else:
            raise ValueError(f"Node '{node_name}' hat keinen unterstützten Increment-Typ")
    
    @property
    def pixel_format(self) -> PixelFormat:
        return PixelFormat.create_from_string_value(
            cast(ids_peak.EnumerationNode, self._get_node("PixelFormat"))
            .CurrentEntry()
            .SymbolicValue()
        )
    
    @pixel_format.setter
    def pixel_format(self, pixel_format: PixelFormat) -> None:
        was_running = self._acquisition_running
        if was_running:
            self.stop_acquisition()
        
        cast(ids_peak.EnumerationNode, self._get_node("PixelFormat")).SetCurrentEntry(
            pixel_format.string_value
        )
        
        if was_running:
            self.start_acquisition()
    
    @property
    def pixel_format_list(self) -> list[PixelFormat]:
        node = cast(ids_peak.EnumerationNode, self._get_node("PixelFormat"))
        excluded = {PixelFormat.RGB_10_PACKED_32, PixelFormat.BGR_10_PACKED_32}
        return [
            PixelFormat.create_from_string_value(e.SymbolicValue())
            for e in node.AvailableEntries()
            if PixelFormat.create_from_string_value(e.SymbolicValue()) not in excluded
        ]
    
    @property
    def exposure(self) -> float:
        return cast(ids_peak.FloatNode, self._get_node("ExposureTime")).Value()
    
    @exposure.setter
    def exposure(self, value: float) -> None:
        cast(ids_peak.FloatNode, self._get_node("ExposureTime")).SetValue(float(value))
    
    @property
    def exposure_range(self) -> Range:
        return self._range_from_node("ExposureTime")
    
    @property
    def framerate(self) -> float:
        return cast(ids_peak.FloatNode, self._get_node("AcquisitionFrameRate")).Value()
    
    @framerate.setter
    def framerate(self, value: float) -> None:
        cast(ids_peak.FloatNode, self._get_node("AcquisitionFrameRate")).SetValue(float(value))
    
    @property
    def framerate_range(self) -> Range:
        return self._range_from_node("AcquisitionFrameRate")
    
    @property
    def has_focus_stepper(self) -> bool:
        node = cast(ids_peak.IntegerNode | None, self._try_get_node("FocusStepper"))
        return node is not None and node.IsAvailable()
    
    @property
    def focus_stepper(self) -> int:
        return cast(ids_peak.IntegerNode, self._get_node("FocusStepper")).Value()
    
    @focus_stepper.setter
    def focus_stepper(self, value: int) -> None:
        cast(ids_peak.IntegerNode, self._get_node("FocusStepper")).SetValue(value)
    
    @property
    def focus_stepper_range(self) -> Range:
        return self._range_from_node("FocusStepper")
    
    def _set_gain_selector(self, gain_type: str) -> None:
        preferred = [f"{prefix}{gain_type}" for prefix in ["Analog", "Digital", ""]]
        
        selector = cast(ids_peak.EnumerationNode | None, self._try_get_node("GainSelector"))
        if selector is None or not selector.IsAvailable():
            return
        
        entries = {e.SymbolicValue() for e in selector.AvailableEntries()}
        
        for pref in preferred:
            if pref in entries:
                selector.SetCurrentEntry(pref)
                return
    
    @property
    def master_gain(self) -> float:
        self._set_gain_selector("All")
        return cast(ids_peak.FloatNode, self._get_node("Gain")).Value()
    
    @master_gain.setter
    def master_gain(self, value: float) -> None:
        self._set_gain_selector("All")
        cast(ids_peak.FloatNode, self._get_node("Gain")).SetValue(value)
    
    @property
    def master_gain_range(self) -> Range:
        self._set_gain_selector("All")
        return self._range_from_node("Gain")
    
    @property
    def red_gain(self) -> float:
        self._set_gain_selector("Red")
        return cast(ids_peak.FloatNode, self._get_node("Gain")).Value()
    
    @red_gain.setter
    def red_gain(self, value: float) -> None:
        self._set_gain_selector("Red")
        cast(ids_peak.FloatNode, self._get_node("Gain")).SetValue(value)
    
    @property
    def red_gain_range(self) -> Range:
        self._set_gain_selector("Red")
        return self._range_from_node("Gain")
    
    @property
    def green_gain(self) -> float:
        self._set_gain_selector("Green")
        return cast(ids_peak.FloatNode, self._get_node("Gain")).Value()
    
    @green_gain.setter
    def green_gain(self, value: float) -> None:
        self._set_gain_selector("Green")
        cast(ids_peak.FloatNode, self._get_node("Gain")).SetValue(value)
    
    @property
    def green_gain_range(self) -> Range:
        self._set_gain_selector("Green")
        return self._range_from_node("Gain")
    
    @property
    def blue_gain(self) -> float:
        self._set_gain_selector("Blue")
        return cast(ids_peak.FloatNode, self._get_node("Gain")).Value()
    
    @blue_gain.setter
    def blue_gain(self, value: float) -> None:
        self._set_gain_selector("Blue")
        cast(ids_peak.FloatNode, self._get_node("Gain")).SetValue(value)
    
    @property
    def blue_gain_range(self) -> Range:
        self._set_gain_selector("Blue")
        return self._range_from_node("Gain")
    
    def gain_type_list(self) -> list[str]:
        selector = cast(ids_peak.EnumerationNode, self._get_node("GainSelector"))
        return [e.SymbolicValue() for e in selector.AvailableEntries()]

    def set_gain(self, gain_type: str, gain: float) -> None:
        cast(ids_peak.EnumerationNode, self._get_node("GainSelector")).SetCurrentEntry(gain_type)
        cast(ids_peak.FloatNode, self._get_node("Gain")).SetValue(float(gain))

    def get_gain(self, gain_type: str) -> tuple[float, Range]:
        cast(ids_peak.EnumerationNode, self._get_node("GainSelector")).SetCurrentEntry(gain_type)
        return (
            cast(ids_peak.FloatNode, self._get_node("Gain")).Value(),
            self._range_from_node("Gain"),
        )

    @property
    def reverse_x(self) -> bool:
        """Gibt aktuellen ReverseX Status zurück (horizontales Spiegeln)."""
        node = cast(ids_peak.BooleanNode | None, self._try_get_node("ReverseX"))
        if node is None or not node.IsAvailable():
            return False
        try:
            return node.Value()
        except CommonException:
            return False

    @reverse_x.setter
    def reverse_x(self, value: bool) -> None:
        """Setzt ReverseX (horizontales Spiegeln)."""
        node = cast(ids_peak.BooleanNode | None, self._try_get_node("ReverseX"))
        if node is None or not node.IsAvailable():
            logger.warning("ReverseX nicht verfügbar")
            return

        try:
            if not node.IsWriteable():
                was_started = self._acquisition_running
                if was_started:
                    self.stop_acquisition()
                node.SetValue(value)
                if was_started:
                    self.start_acquisition()
            else:
                node.SetValue(value)
            self._reverse_x = value
            logger.info(f"ReverseX gesetzt: {value}")
        except CommonException as e:
            logger.warning(f"ReverseX konnte nicht gesetzt werden: {e}")

    @property
    def reverse_y(self) -> bool:
        """Gibt aktuellen ReverseY Status zurück (vertikales Spiegeln)."""
        node = cast(ids_peak.BooleanNode | None, self._try_get_node("ReverseY"))
        if node is None or not node.IsAvailable():
            return False
        try:
            return node.Value()
        except CommonException:
            return False

    @reverse_y.setter
    def reverse_y(self, value: bool) -> None:
        """Setzt ReverseY (vertikales Spiegeln)."""
        node = cast(ids_peak.BooleanNode | None, self._try_get_node("ReverseY"))
        if node is None or not node.IsAvailable():
            logger.warning("ReverseY nicht verfügbar")
            return

        try:
            if not node.IsWriteable():
                was_started = self._acquisition_running
                if was_started:
                    self.stop_acquisition()
                node.SetValue(value)
                if was_started:
                    self.start_acquisition()
            else:
                node.SetValue(value)
            self._reverse_y = value
            logger.info(f"ReverseY gesetzt: {value}")
        except CommonException as e:
            logger.warning(f"ReverseY konnte nicht gesetzt werden: {e}")
    
    def reset_to_default(self) -> None:
        """Setzt Kamera auf Default-UserSet zurück."""
        self.stop_acquisition()
        self._node_cache.clear()
        self._load_userset("Default")
        self._fix_coordinates()
        self._disable_device_autofeatures()
        self.start_acquisition()
    
    def save_userset(self, userset: str = "UserSet0") -> None:
        """Speichert aktuelle Einstellungen in UserSet."""
        selector = cast(ids_peak.EnumerationNode, self._get_node("UserSetSelector"))
        available = [e.SymbolicValue() for e in selector.AvailableEntries()]
        
        if userset not in available:
            raise ValueError(f"UserSet '{userset}' nicht verfügbar. Optionen: {available}")
        
        selector.SetCurrentEntry(userset)
        cast(ids_peak.CommandNode, self._get_node("UserSetSave")).Execute()
        logger.info(f"UserSet '{userset}' gespeichert")
    
    def configure_and_save(
        self,
        userset: str = "UserSet0",
        exposure: float | None = None,
        framerate: float | None = None,
        gain: float | None = None,
        pixel_format: PixelFormat | None = None,
        width: int | None = None,
        height: int | None = None,
        offset_x: int | None = None,
        offset_y: int | None = None,
    ) -> None:
        """
        Konfiguriert Kamera-Einstellungen und speichert sie in UserSet.
        
        Args:
            userset: Ziel-UserSet ("UserSet0", "UserSet1", etc.)
            exposure: Belichtungszeit in µs
            framerate: Bildrate in fps
            gain: Master-Gain
            pixel_format: Pixel-Format (z.B. PixelFormat.RGB_8)
            width: Bildbreite (ROI)
            height: Bildhöhe (ROI)
            offset_x: ROI X-Offset
            offset_y: ROI Y-Offset
        
        Raises:
            ValueError: Wenn UserSet nicht verfügbar
            CommonException: Wenn Speichern fehlschlägt (z.B. schreibgeschützt)
        """
        was_running = self._acquisition_running
        if was_running:
            self.stop_acquisition()
        
        try:
            # ROI zuerst (beeinflusst andere Parameter)
            if width is not None:
                cast(ids_peak.IntegerNode, self._get_node("Width")).SetValue(width)
                logger.info(f"Width gesetzt: {width}")
            
            if height is not None:
                cast(ids_peak.IntegerNode, self._get_node("Height")).SetValue(height)
                logger.info(f"Height gesetzt: {height}")
            
            if offset_x is not None:
                cast(ids_peak.IntegerNode, self._get_node("OffsetX")).SetValue(offset_x)
                logger.info(f"OffsetX gesetzt: {offset_x}")
            
            if offset_y is not None:
                cast(ids_peak.IntegerNode, self._get_node("OffsetY")).SetValue(offset_y)
                logger.info(f"OffsetY gesetzt: {offset_y}")
            
            if pixel_format is not None:
                self.pixel_format = pixel_format
                logger.info(f"Pixel Format gesetzt: {pixel_format}")
            
            if exposure is not None:
                self.exposure = exposure
                logger.info(f"Exposure gesetzt: {exposure:.1f} µs")
            
            if framerate is not None:
                self.framerate = framerate
                logger.info(f"Framerate gesetzt: {framerate:.2f} fps")
            
            if gain is not None:
                self.master_gain = gain
                logger.info(f"Master Gain gesetzt: {gain:.2f}")
            
            # In UserSet speichern
            self.save_userset(userset)
            
            # Neue Einstellungen ausgeben
            self._log_current_settings()
            
        finally:
            if was_running:
                self.start_acquisition()
    
    def get_current_settings(self) -> dict:
        """
        Gibt aktuelle Kamera-Einstellungen als Dictionary zurück.
        
        Returns:
            Dict mit allen relevanten Einstellungen
        """
        settings = {
            "pixel_format": str(self.pixel_format),
            "width": self._get_width(),
            "height": self._get_height(),
            "exposure_us": self.exposure,
            "framerate_fps": self.framerate,
            "master_gain": self.master_gain,
        }
        
        # ROI Offsets
        try:
            settings["offset_x"] = cast(ids_peak.IntegerNode, self._get_node("OffsetX")).Value()
            settings["offset_y"] = cast(ids_peak.IntegerNode, self._get_node("OffsetY")).Value()
        except CommonException:
            pass
        
        # RGB Gains
        try:
            settings["red_gain"] = self.red_gain
            settings["green_gain"] = self.green_gain
            settings["blue_gain"] = self.blue_gain
        except CommonException:
            pass
        
        # Exposure Range für Referenz
        exp_range = self.exposure_range
        settings["exposure_range"] = {
            "min": exp_range.minimum,
            "max": exp_range.maximum,
        }
        
        # Framerate Range
        fps_range = self.framerate_range
        settings["framerate_range"] = {
            "min": fps_range.minimum,
            "max": fps_range.maximum,
        }

        # Image orientation
        settings["reverse_x"] = self.reverse_x
        settings["reverse_y"] = self.reverse_y

        return settings
    
    def load_settings_file(self, filepath: str) -> None:
        """
        Lädt Kamera-Einstellungen aus einer .cset Datei (GenApi Persistence File).
        
        Args:
            filepath: Pfad zur .cset Datei
            
        Raises:
            FileNotFoundError: Wenn Datei nicht existiert
            CommonException: Wenn Laden fehlschlägt
        """
        import os
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Settings-Datei nicht gefunden: {filepath}")
        
        was_running = self._acquisition_running
        if was_running:
            self.stop_acquisition()
        
        try:
            # Node-Cache invalidieren (Einstellungen ändern sich)
            self._node_cache.clear()
            
            # GenApi LoadFromFile nutzen
            self.remote_device_nodemap.LoadFromFile(filepath)
            logger.info(f"Einstellungen geladen aus: {filepath}")
            
            # Neue Einstellungen ausgeben
            self._log_current_settings()
            
        finally:
            if was_running:
                self.start_acquisition()
    
    def save_settings_file(self, filepath: str) -> None:
        """
        Speichert aktuelle Kamera-Einstellungen in eine .cset Datei.
        
        Args:
            filepath: Ziel-Pfad für die .cset Datei
        """
        self.remote_device_nodemap.StoreToFile(filepath)
        logger.info(f"Einstellungen gespeichert nach: {filepath}")
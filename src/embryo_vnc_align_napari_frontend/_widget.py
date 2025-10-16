from typing import TYPE_CHECKING, Callable, Sequence
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from dataclasses import dataclass
import re

from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
from pyometiff import OMETIFFWriter

from qtpy.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QLineEdit,
    QWidget,
    QComboBox,
    QHBoxLayout,
    QLabel,
)
from qtpy.QtGui import QIntValidator, QDoubleValidator
from qt_remote_commands_over_ssh_for_napari_plugins import (
    add_widgets,
    Client,
    to_string,
    raise_exception,
)
from napari.layers import Image, Points
from napari.qt.threading import thread_worker, FunctionWorker
import numpy as np
from tifffile import TiffFile, TiffFrame

logging.basicConfig(
    filename="app.log",
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


AddImageKwargs = dict[
    str, np.ndarray | tuple[float, float, float] | str | float | list[str]
]


@dataclass(frozen=True)
class RotationIdentificationRequest:
    coords: list[list[float]]
    scale: list[float]
    index: int
    input_path: str
    pixel_buffer_factor: float
    height: float


@dataclass(frozen=True)
class ApplyRotationRequest:
    input_path: str
    index: int
    pixel_buffer_factor: float
    height: float


if TYPE_CHECKING:
    import napari
    import napari.viewer
    from napari.utils.events import Event


def image_args_from_path(path: Path) -> AddImageKwargs:
    """
    reads the ometif
    """
    with TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.get_axes()
        assert axes in ["CZYX", "ZYX"]
        first_page = series.pages[0]
        if first_page is None:
            raise ValueError("Could not read resolution")
        if isinstance(first_page, TiffFrame):
            raise ValueError("Could not read resolution")
        assert tif.ome_metadata is not None
        mdata = ET.fromstring(tif.ome_metadata)
        image = mdata[0]
        namespace = {"ome": next(iter(mdata.attrib.values())).split()[0]}
        pixels = image.find("ome:Pixels", namespaces=namespace)
        channel_names = [
            ch.attrib["Name"]
            for ch in mdata.findall(".//ome:Channel", namespace)
        ]
        assert pixels is not None
        scale = (
            float(pixels.get("PhysicalSizeZ", 0)),
            float(pixels.get("PhysicalSizeY", 0)),
            float(pixels.get("PhysicalSizeX", 0)),
        )
        data = series.asarray()
        if "C" in axes:
            return {
                "data": data,
                "scale": scale,
                "channel_axis": 0,
                "name": channel_names[0],
            }
        return {"data": data, "scale": scale, "name": channel_names}


def get_points_name_callback(
    labels: Sequence[str],
) -> "Callable[[Event], None]":
    def callback(event: "Event"):
        if event.action == "adding":
            logger.debug("skipping added")
            return
        layer: Points = event.source
        n_points = len(layer.data)
        # ensure that extra points are unlabeled
        extra_labels = list(labels) + [""]
        if n_points <= len(extra_labels):
            layer.properties["text"][:] = extra_labels[: len(layer.data)]
            layer.refresh_text()
            logger.debug(layer.properties["text"])

    return callback


class RotateVnc(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        viewer.layers.events.inserted.connect(self.reset_image_box)
        viewer.layers.events.removed.connect(self.reset_image_box)
        layout = QVBoxLayout()
        self.setLayout(layout)
        self.cm = add_widgets(layout, print)
        image_box_row = QHBoxLayout()
        image_box_row.addWidget(QLabel("Reference channel"))
        self.image_box = QComboBox()
        image_box_row.addWidget(self.image_box)
        layout.addLayout(image_box_row)
        pix_bf_row = QHBoxLayout()
        pix_bf_row.addWidget(QLabel("Pixel buffer factor"))
        self.pix_bf = QLineEdit()
        self.pix_bf.setValidator(QDoubleValidator())
        self.pix_bf.setText("1")
        pix_bf_row.addWidget(self.pix_bf)
        layout.addLayout(pix_bf_row)
        vnc_depth_row = QHBoxLayout()
        vnc_depth_row.addWidget(QLabel("vnc_depth"))
        self.vnc_depth = QLineEdit()
        self.vnc_depth.setValidator(QDoubleValidator())
        self.vnc_depth.setText("40")
        vnc_depth_row.addWidget(self.vnc_depth)
        layout.addLayout(vnc_depth_row)
        scene_idx_row = QHBoxLayout()
        scene_idx_row.addWidget(QLabel("Scene index"))
        self.scene_idx = QLineEdit()
        self.scene_idx.setValidator(QIntValidator(bottom=0, top=999))
        self.scene_idx.setText("0")
        scene_idx_row.addWidget(self.scene_idx)
        layout.addLayout(scene_idx_row)
        self.submit_button = QPushButton("Submit Coords")
        self.submit_button.clicked.connect(
            self.send_rotation_identification_request
        )
        layout.addWidget(self.submit_button)
        self.rotate_button = QPushButton("Rotate Image")
        self.rotate_button.clicked.connect(self.send_apply_rotation_request)
        layout.addWidget(self.rotate_button)

        try:
            self.reference_layer = next(
                l for l in viewer.layers if isinstance(l, Image)
            )
        except StopIteration:
            self.reference_layer = Image(data=np.zeros((1, 1, 1)))
        self.anterior_posterior = viewer.add_points(
            data=[],
            ndim=3,
            scale=self.reference_layer.scale,
            name="anterior posterior",
            properties={"text": np.array([]).astype(str)},
            text="text",
        )
        self.anterior_posterior.mode = "add"
        self.anterior_posterior.events.data.connect(
            get_points_name_callback(["anterior", "posterior"])
        )
        self.lateral = viewer.add_points(
            data=[],
            ndim=3,
            scale=self.reference_layer.scale,
            name="left right",
            properties={"text": np.array([]).astype(str)},
            text="text",
        )
        self.lateral.mode = "add"
        self.lateral.events.data.connect(
            get_points_name_callback(["side", "side"])
        )
        self._client: Client | None = None

        self.cm.host_name.setText("localhost")
        assert isinstance(self.cm.exe, QLineEdit)
        self.cm.exe.setText("bin/align-server")

    def reset_image_box(self):
        """
        resets a combo box to a new set of values
        """
        old_value = self.image_box.currentText()
        self.image_box.clear()
        # avoid repeat labels
        values = set(
            l.name for l in self.viewer.layers if isinstance(l, Image)
        )
        self.image_box.addItems(list(values))
        if old_value in values:
            self.image_box.setCurrentText(old_value)
        final_value = self.image_box.currentText()
        if final_value:
            scene_or_none = self.viewer.layers[final_value].metadata.get(
                "scene_index"
            )
            if scene_or_none is not None:
                self.scene_idx.setText(str(scene_or_none))

    def get_coords(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        return anterior, posterior, side, side
        """
        anterior, posterior = self.anterior_posterior.data[:2]
        s1, s2 = self.lateral.data[:2]
        return anterior, s1, s2, posterior

    @thread_worker
    def send_rotation_identification_request_thread(
        self,
    ) -> AddImageKwargs:
        """
        In a background thread, create a file, send it over ssh, connect to the
        client, send a rotation send_rotation_identification_request. returns the args to add_image
        """
        # create a file
        data = self.viewer.layers[self.image_box.currentText()].data
        in_scale = self.viewer.layers[self.image_box.currentText()].scale
        out_scale = 0.5
        zoom_values = in_scale / out_scale
        zoomed_data = ndi.zoom(data, zoom_values, order=0)

        in_path = Path("reference.npy")
        np.save(in_path, img_as_ubyte(zoomed_data))
        # write to the server
        coords = [c.tolist() for c in self.get_coords()]
        rir = RotationIdentificationRequest(
            coords=(coords * in_scale / out_scale).tolist(),  # type: ignore
            scale=[out_scale] * 3,
            index=int(self.scene_idx.text()),
            input_path=str(in_path),
            pixel_buffer_factor=float(self.pix_bf.text()),
            height=float(self.vnc_depth.text()),
        )
        try:
            with self.cm as client:
                client.send_file(in_path)
                response = client.request(to_string(rir), timeout=9999)
                if response.error:
                    raise RuntimeError(response.error)
        finally:
            in_path.unlink(missing_ok=True)
        file = Path(response.out)
        client.receive_file(file, file)
        logger.info(response)
        try:
            out = image_args_from_path(file) | {"name": "preview"}
        finally:
            file.unlink(missing_ok=True)
        return out

    def post_rotation_identification_request(self, kwargs: AddImageKwargs):
        self.submit_button.setChecked(False)
        self.submit_button.setEnabled(True)
        viewer.add_image(**kwargs)

    def send_rotation_identification_request(self, *args):
        self.submit_button.setChecked(True)
        self.submit_button.setEnabled(False)
        _ = args
        # verify layers
        try:
            _ = self.get_coords()
        except ValueError:
            print("You must populate layers")
            return
        worker: FunctionWorker = self.send_rotation_identification_request_thread()  # type: ignore
        worker.returned.connect(self.post_rotation_identification_request)
        worker.errored.connect(raise_exception)
        worker.start()

    @thread_worker
    def send_apply_rotation_request_thread(self) -> AddImageKwargs:
        regex_pattern = re.compile(r"^raw-(.+)-channel$")
        regex_match = re.match(regex_pattern, self.reference_layer.name)
        if regex_match is None:
            channel_name_map = {
                l.name: l for l in self.viewer.layers if isinstance(l, Image)
            }
        else:
            channel_name_map = {}
            for layer in self.viewer.layers:
                if not isinstance(layer, Image):
                    continue
                layer_match = re.match(regex_pattern, layer.name)
                if layer_match is None:
                    continue
                channel_name_map[layer.name] = layer
        channel_md: dict[str, dict] = {}
        channel_data: list[np.ndarray] = []
        for name, channel in channel_name_map.items():
            channel_data.append(np.array(channel.data))
            channel_md[name] = {"Name": name}
        scale = next(iter(channel_name_map.values())).scale
        metadata_dict = {
            "PhysicalSizeX": scale[2],
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": scale[1],
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": scale[0],
            "PhysicalSizeZUnit": "µm",
            "Channels": channel_md,
        }
        array = np.stack(channel_data)
        multi_chan_path = Path(f"S{self.scene_idx.text()}.ome.tiff")
        writer = OMETIFFWriter(
            fpath=multi_chan_path,
            dimension_order="CZYX",
            array=array,
            metadata=metadata_dict,
        )
        logger.debug("Composing OME tiff")
        writer.write()
        arr = ApplyRotationRequest(
            input_path=str(multi_chan_path),
            index=int(self.scene_idx.text()),
            pixel_buffer_factor=float(self.pix_bf.text()),
            height=40,
        )
        # send file
        try:
            with self.cm as client:
                client.send_file(multi_chan_path)
                # copy over landmarks
                client.remote_cp(
                    Path(f"reference-S{self.scene_idx.text()}.landmarks"),
                    Path(f"S{self.scene_idx.text()}.landmarks"),
                )
                response = client.request(to_string(arr), 9999)
                if response.error:
                    raise RuntimeError(response.error)
        finally:
            multi_chan_path.unlink(missing_ok=True)
        file = Path(response.out)
        client.receive_file(file, file)
        logger.info(response)
        try:
            out = image_args_from_path(file)
        finally:
            file.unlink(missing_ok=True)
        return out

    def post_rotation_request(self, kwargs: AddImageKwargs):
        self.rotate_button.setChecked(False)
        self.rotate_button.setEnabled(True)
        self.viewer.add_image(**kwargs)

    def send_apply_rotation_request(self, arg):
        self.rotate_button.setChecked(True)
        self.rotate_button.setEnabled(False)
        _ = arg
        if "preview" not in self.viewer.layers[-1].name:
            print("you must first submit coords")
        # convert image into ometif
        worker: FunctionWorker = self.send_apply_rotation_request_thread()  # type: ignore
        worker.returned.connect(self.post_rotation_request)

        worker.errored.connect(raise_exception)
        worker.start()

    def closeEvent(self, a0):
        """Clean up client connection when widget is closed"""
        if self.cm._client:
            self.cm._client.close()
        super().closeEvent(a0)


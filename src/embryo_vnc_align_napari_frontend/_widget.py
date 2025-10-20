from typing import TYPE_CHECKING, Callable, Sequence, Generator
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
    QWidget,
)
from qt_remote_commands_over_ssh_for_napari_plugins import (
    to_string,
)
from qt_remote_commands_over_ssh_for_napari_plugins.client import (
    ConnectionManager,
    GuiBackgroundFunction,
    Argument,
)
from napari.layers import Image, Points
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
        layout = QVBoxLayout()
        self.setLayout(layout)
        # load connection widgets
        self.cm = ConnectionManager.create(
            print,
            "localhost",
            "bin/align-server",
        )
        self.cm.get_gui_background_function().add_widgets(layout)
        # Create get_gui_background_function for remote_gamma
        self.submitted_coords = False
        self.submit_coords = GuiBackgroundFunction[AddImageKwargs].create(
            "Submit coords",
            self.send_rotation_identification_request_thread,
            self.add_image,
            arguments=(
                Argument(
                    "Reference channel",
                    "Channel that will be transformed ",
                    Image,
                    None,
                ),
                Argument(
                    "Pixel buffer factor",
                    "Higher numbers increases the ammount of medial and anterior / posterior buffer included in the image",
                    float,
                    1.0,
                ),
                Argument(
                    "VNC depth",
                    "The final dorsal / ventral buffer in the image in microns",
                    float,
                    40.0,
                ),
                Argument(
                    "Scene index",
                    "An itentifying index of the scene (perhaps of the czi file)",
                    int,
                    0,
                ),
            ),
            viewer=self.viewer,
        )
        self.submit_coords.add_widgets(layout)
        self.rotate_image = GuiBackgroundFunction[AddImageKwargs].create(
            "Apply rotation",
            self.send_apply_rotation_request_thread,
            self.add_image,
            arguments=tuple(),
            viewer=self.viewer,
        )
        self.rotate_image.add_widgets(layout)

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

    def get_coords(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        return anterior, posterior, side, side
        """
        anterior, posterior = self.anterior_posterior.data[:2]
        s1, s2 = self.lateral.data[:2]
        return anterior, s1, s2, posterior

    def send_rotation_identification_request_thread(
        self, ref_chan: Image, pix_bf: float, depth: float, scene_idx: int
    ) -> Generator[str, None, AddImageKwargs]:
        """
        In a background thread, create a file, send it over ssh, connect to the
        client, send a rotation send_rotation_identification_request. returns the args to add_image
        """
        # create a file
        yield "saving the file"
        data = ref_chan.data
        in_scale = ref_chan.scale
        out_scale = 0.5
        zoom_values = in_scale / out_scale
        zoomed_data = ndi.zoom(data, zoom_values, order=0)

        # write to the server
        coords = [c.tolist() for c in self.get_coords()]
        in_path = Path("reference.npy")
        np.save(in_path, img_as_ubyte(zoomed_data))
        rir = RotationIdentificationRequest(
            coords=(coords * in_scale / out_scale).tolist(),  # type: ignore
            scale=[out_scale] * 3,
            index=scene_idx,
            input_path=str(in_path),
            pixel_buffer_factor=pix_bf,
            height=depth,
        )
        try:
            with self.cm as client:
                yield "sending the file"
                client.send_file(in_path)
                yield "processing the data"
                response = client.request(to_string(rir), timeout=9999)
                if response.error:
                    raise RuntimeError(response.error)
        finally:
            in_path.unlink(missing_ok=True)
        file = Path(response.out)
        yield "receiving the file"
        client.receive_file(file, file)
        logger.info(response)
        try:
            out = image_args_from_path(file) | {"name": "preview"}
        finally:
            file.unlink(missing_ok=True)
        yield "Done"
        self.submitted_coords = True
        return out

    def add_image(self, kwargs: AddImageKwargs):
        """
        runs in main thread after eigher thread worker
        """
        self.viewer.add_image(**kwargs)  # type: ignore

    def send_apply_rotation_request_thread(
        self,
    ) -> Generator[str, None, AddImageKwargs]:
        if not self.submitted_coords:
            raise RuntimeError("First submit coords")
        yield "writing file"
        regex_pattern = re.compile(r"^raw-(.+)-channel$")
        reference_layer, pix_bf, depth, scene_idx = (
            self.submit_coords.get_values()
        )
        regex_match = re.match(regex_pattern, reference_layer.name)
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
        multi_chan_path = Path(f"S{scene_idx}.ome.tiff")
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
            index=scene_idx,
            pixel_buffer_factor=pix_bf,
            height=depth,
        )
        # send file
        try:
            with self.cm as client:
                yield "sending file"
                client.send_file(multi_chan_path)
                # copy over landmarks
                client.remote_cp(
                    Path(f"reference-S{scene_idx}.landmarks"),
                    Path(f"S{scene_idx}.landmarks"),
                )
                yield "processing data"
                response = client.request(to_string(arr), 9999)
                if response.error:
                    raise RuntimeError(response.error)
        finally:
            multi_chan_path.unlink(missing_ok=True)
        file = Path(response.out)
        yield "receive file"
        client.receive_file(file, file)
        logger.info(response)
        try:
            out = image_args_from_path(file)
        finally:
            file.unlink(missing_ok=True)
        yield "Done"
        return out

    def closeEvent(self, a0):
        """Clean up client connection when widget is closed"""
        if self.cm._client:
            self.cm._client.close()
        super().closeEvent(a0)

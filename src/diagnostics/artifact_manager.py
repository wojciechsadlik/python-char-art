import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from PIL import Image, ImageFont

from rendering.image import render_symbols_img

logger = logging.getLogger(__name__)


class ArtifactManager:
    def __init__(
        self,
        base_dir: str = "outputs",
        run_name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.current_line: int = 0
        self.current_gen: int = 0

        if self.debug:
            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            self.run_dir = Path(base_dir) / run_name
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def set_line(self, line_idx: int, src_line: Optional[Image.Image] = None) -> None:
        self.current_line = line_idx

        if not self.debug or src_line is None:
            return

        line_dir = self.run_dir / f"line_{self.current_line:03d}"
        line_dir.mkdir(exist_ok=True)

        src_path = line_dir / "src.png"
        if not src_path.exists():
            src_line.save(src_path)
            logger.debug("Saved source line snippet: %s", src_path)

    def save_line(
        self,
        symbol_line: list[str],
        font: ImageFont.FreeTypeFont,
        size: tuple[int, int],
        fitness: float,
    ) -> Optional[Path]:
        if not self.debug:
            return None

        line_dir = self.run_dir / f"line_{self.current_line:03d}"
        line_dir.mkdir(exist_ok=True)

        res_img = render_symbols_img([symbol_line], font, wh=size)

        filename = f"gen_{self.current_gen:04d}_fit_{fitness:.4f}.png"
        filepath = line_dir / filename
        res_img.save(filepath)
        logger.debug("Saved state artifact: %s", filepath)
        return filepath


_instance: Optional[ArtifactManager] = None


def get_artifact_manager(
    base_dir: str = "outputs",
    run_name: Optional[str] = None,
    debug: bool = False,
    reset: bool = False,
) -> ArtifactManager:
    global _instance
    if _instance is None or reset:
        _instance = ArtifactManager(base_dir=base_dir, run_name=run_name, debug=debug)
    return _instance

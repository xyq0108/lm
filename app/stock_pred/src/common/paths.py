"""路径管理"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectPaths:
    """统一的项目目录结构"""
    data_dir: str = "./data"
    model_dir: str = "./model"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    report_dir: str = "./reports"

    def __post_init__(self):
        self.data_dir = str(Path(self.data_dir).resolve())
        self.model_dir = str(Path(self.model_dir).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())
        self.temp_dir = str(Path(self.temp_dir).resolve())
        self.report_dir = str(Path(self.report_dir).resolve())

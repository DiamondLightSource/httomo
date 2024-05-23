import importlib.machinery
import importlib.util
from pathlib import Path

from setuptools import setup, Command
from setuptools.command import build


class GenerateYamlTemplates(Command):
    def run(self) -> None:
        # Create `yaml_templates/` dir and necessary subdirs in the `build/` dir
        BACKENDS = ["httomolib", "httomolibgpu", "tomopy"]
        yaml_templates_dir = Path(__file__).parent / "build/lib/httomo/yaml_templates"
        yaml_templates_dir.mkdir(exist_ok=True)
        backends_template_dirs = [
            Path(yaml_templates_dir / backend) for backend in BACKENDS
        ]
        for backend_dir in backends_template_dirs:
            backend_dir.mkdir(exist_ok=True)

        # Load the `yaml_templates_generator.py` script as a module
        loader = importlib.machinery.SourceFileLoader(
            "yaml_generator",
            str(Path(__file__).parent / "scripts/yaml_templates_generator.py"),
        )
        spec = importlib.util.spec_from_loader("yaml_generator", loader)
        assert spec is not None
        yaml_templates_generator = importlib.util.module_from_spec(spec)
        loader.exec_module(yaml_templates_generator)

        # Get paths to backends + httomo dirs in methods database, to get the YAML file
        # containing list of methods from each backend
        methods_database_dir = (
            Path(__file__).parent / "httomo/methods_database/packages/external"
        )
        backends_yaml_lists = [
            methods_database_dir / f"{backend}/{backend}_modules.yaml"
            for backend in BACKENDS
        ]

        # Generate YAML templates for each backend
        for yaml_list, backend_dir in zip(backends_yaml_lists, backends_template_dirs):
            yaml_templates_generator.yaml_generator(yaml_list, backend_dir)

    def initialize_options(self): ...

    def finalize_options(self): ...


build.build.sub_commands.append(("generate_yaml_templates", None))

setup(
    cmdclass={
        "generate_yaml_templates": GenerateYamlTemplates,
    }
)

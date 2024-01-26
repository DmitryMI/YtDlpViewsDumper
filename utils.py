import importlib
import logging
import os
import pkgutil
import re

logger = logging.getLogger("Utils")


def find_subclasses(clazz, directory: str, tag_field_name: str):
    package = os.path.basename(directory)
    # directory = f"../{package}"
    for (module_loader, name, is_package) in pkgutil.iter_modules([directory]):
        logger.info(f"Importing {name}...")
        importlib.import_module('.' + name, package)

    subclasses = {cls.__name__: cls for cls in clazz.__subclasses__()}

    for name, clazz in subclasses.items():
        if tag_field_name in clazz.__dict__:
            yield clazz


def find_subclasses_with_tags(class_type: type, directory: str, tag_field_name: str, tags: str | list[str]):
    if tags is None:
        return
    if type(tags) is str:
        tags = [tags]

    clazzes = find_subclasses(class_type, directory, tag_field_name)
    for clazz in clazzes:
        d = clazz.__dict__
        tag_value = d[tag_field_name]
        if tag_value in tags:
            yield clazz


def find_subclasses_tags_regex(class_type: type, directory: str, tag_field_name: str, tags: str | list[str]):
    if tags is None:
        return
    if type(tags) is str:
        tags = [tags]

    clazzes = find_subclasses(class_type, directory, tag_field_name)
    for clazz in clazzes:
        d = clazz.__dict__
        tag_regex = d[tag_field_name]
        if tag_regex is None:
            continue

        for tag in tags:
            if re.match(tag_regex, tag):
                yield clazz

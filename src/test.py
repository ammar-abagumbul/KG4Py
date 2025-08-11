import json
from compact_json import CompactJSONEncoder

test_data_with_lists = {
    "name": "example",
    "version": "1.0",
    "dependencies": ["lib1", "lib2"],
    "metadata": {"authors": ["Alice", "Bob"], "license": "MIT"},
    "numbers": [
        1.0,
        2.5,
        3.14,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
    ],
}

with open("test_output.json", "w", encoding="utf-8") as f:
    json.dump(test_data_with_lists, f, indent=4, cls=CompactJSONEncoder)

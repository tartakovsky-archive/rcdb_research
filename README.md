## Installation/upgrade from inside of Jupyter

```bash
$ pip install -U git+https://github.com/tartakovsky-archive/rcdb_research
```

## Installation for development

`$ pip install .` - install requirements from source  
`$ pip install -e .[dev]` - install requirements from source for development  
`$ pip install -e <git url>` - install requirements from git  

For installation another branch into current pip use environment variable `DEV_PREFIX` with prefix name 
e.g. install lib from `dev` branch to `development_rcdb_research`:
```bash
$ DEV_PREFIX=development pip install -U --extra-index-url https://pypi-private:***TOKEN***@pkgs.dev.azure.com/rcdb/_packaging/pypi-private/pypi/simple/ git+https://github.com/tartakovsky-archive/rcdb_research@dev
```

## Custom constraints for config parameters
```python
test_config = dict(
        f=[
            dict(
                fn=inc_func,
                pg=km(a=[1, 2, 3], b=[1, 2, 3], c=[1, -1]),
                dm=km(x=['input']),
                cn='p.a < p.b and p.c != -1'
            )
        ]
    )
```

Result parameters sets:
```python
[
    dict(a=1, b=2, c=1),
    dict(a=1, b=3, c=1),
    dict(a=2, b=3, c=1),
]
```

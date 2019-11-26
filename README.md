# rcdb_research

## Dependencies

Install numpy:
```bash
$ pip install numpy==1.16.4
```  

## Installation/upgrade from inside of Jupyter

Run the following in the terminal, you'll be prompted for github credentials:

```bash
$ pip install -U --extra-index-url https://pypi-private:***TOKEN***@pkgs.dev.azure.com/rcdb/_packaging/pypi-private/pypi/simple/ git+https://github.com/tartakovsky-archive/rcdb_research
```

## Installation for development

`$ pip install --extra-index-url $(cat extra-index-url) .` - install requirements from source  
`$ pip install --extra-index-url $(cat extra-index-url) -e '.[dev]'` - install requirements from source for development  
`$ pip install --extra-index-url $(cat extra-index-url) -e <git url>` - install requirements from git  
`$ jupyter notebook` - start jupyter  

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

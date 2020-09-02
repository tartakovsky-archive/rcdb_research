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
$ DEV_PREFIX=development pip install -U git+https://github.com/tartakovsky-archive/rcdb_research@dev
```

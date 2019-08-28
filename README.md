# research-notebooks

## Dependencies
First install [ta-lib`s dependencies](http://mrjbq7.github.io/ta-lib/install.html):
```bash
$ chomd +x install_talib.sh
$ sudo ./install_talib.sh
```  

Install numpy:
```bash
$ pip install numpy==1.16.4
```  

## Installation/upgrade from inside of Jupyter

Run the following in the terminal, you'll be prompted for github credentials:
```bash
$ $(curl https://gist.githubusercontent.com//22e066cb42b09c6cc89ca7edb881252f/raw/813658c3a93873b7a35971465f8a2ac3be56c81c/dev-feature-lib)
```

Install for default envirnoment (`--user` flag):

```bash
$ $(echo $(curl https://gist.githubusercontent.com//22e066cb42b09c6cc89ca7edb881252f/raw/813658c3a93873b7a35971465f8a2ac3be56c81c/dev-feature-lib)" --user")
```


```bash
$ pip install -U --user --extra-index-url $(cat extra-index-url) git+https://github.com/tartakovsky-archive/research-notebooks
```

## Installation for development

`$ pip install --extra-index-url $(cat extra-index-url) .` - install requirements from source  
`$ pip install --extra-index-url $(cat extra-index-url) -e '.[dev]'` - install requirements from source for development  
`$ pip install --extra-index-url $(cat extra-index-url) -e <git url>` - install requirements from git  
`$ jupyter notebook` - start jupyter  

See examples at `./notebooks/alex_o/`

## Consolidators
The `commons.bars` is a module that contains consolidators 
set for creation trade bars of different types. 
```python
# Bars that contains volume above 500 btc.
df = bars.volume.fixed(btc_usd, 500, by_quote=False)
```

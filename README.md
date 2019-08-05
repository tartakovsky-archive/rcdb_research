## Installing

`# pip install --extra-index-url $(cat extra-index-url) .` - install requirements from source  
`# pip install --extra-index-url $(cat extra-index-url) -e .[dev]` - install requirements from source for developing  
`# pip install --extra-index-url $(cat extra-index-url) -e <git url>` - install requirements from git  
`# jupyter notebook` - start jupyter  

See examples at `./notebooks/alex_o/`

## Consolidators
The `commons.bars` is a module that contains consolidators 
set for creation trade bars of different types. 
```python
# Bars that contains volume above 500 btc.
df = bars.volume.fixed(btc_usd, 500, by_quote=False)
```

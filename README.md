##Installing

`# pip install -e .` - install requirements  
`# jupyter notebook` - start jupyter  

See examples at `./notebooks/alex_o/`

## Consolidators
The `commons.bars` is a module that contains consolidators 
set for creation trade bars of different types. 
```python
# Bars that contains volume above 500 btc.
df = bars.volume.fixed(btc_usd, 500, by_quote=False)
```


### Known Issues

1. On macOS Big Sur OpenGL is not found see discussion on [StackOverflow for details](https://stackoverflow.com/questions/63475461/unable-to-import-opengl-gl-in-python-on-macos)
  - Temporary Fix: 
    + Find virtual environment path `poetry show -v`
    + Open `<venvpath>/lib/OpenGL/platform/ctypesloader.py`
    + Replace line `fullName = util.find_library( name )`
    + With line `fullName = "/System/Library/Frameworks/{}.framework/{}".format(name,name)`

### Credits 
1. [Hack Font](https://github.com/source-foundry/Hack)
2. [Gidole Font](https://github.com/larsenwork/Gidole)

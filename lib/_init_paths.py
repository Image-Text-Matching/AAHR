import os.path as osp
import sys
# 将指定路径添加到Python解释器的模块搜索路径中
# 当前脚本的父目录下的lib目录添加到Python解释器的模块搜索路径中。这样做可以让Python解释器能够在运行时找到并加载lib目录下的模块
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


root_dir = osp.abspath(osp.dirname(osp.join(__file__, '..')))

lib_path = osp.join(root_dir, 'lib')
add_path(lib_path)


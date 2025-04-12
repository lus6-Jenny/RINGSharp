import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0'
    write_version_to_file(version, 'glnet/version.py')

    setup(
        name='glnet',
        version=version,
        description='A general codebase for global localization networks',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'SharedArray',
            # 'spconv',  # spconv has different names depending on the cuda version
        ],

        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='voxel_pooling_train_ext',
                module='glnet.ops.voxel_pooling_train',
                sources=[
                    'src/voxel_pooling_train_forward.cpp',
                    'src/voxel_pooling_train_forward_cuda.cu'],
            ),
            make_cuda_ext(
                name='voxel_pooling_inference_ext',
                module='glnet.ops.voxel_pooling_inference',
                sources=[
                    'src/voxel_pooling_inference_forward.cpp',
                    'src/voxel_pooling_inference_forward_cuda.cu'],
            ),            
        ],
    )

# Run
```python
python ./main.py
```

# Build -> EXE
```python
pyinstaller ./build.spec
```

# Client Application Instructions
1. 注意路径选择中，不要含有中文和空格。
2. 以第一帧的GPS位置为参考系，选择高程。（第一帧位置往往位于地面，因此地面位置高程为0，故只需要在-20~50m进行高程选择即可。高程范围选择是为了屏蔽掉无人机升降导致的点云噪声）
3. 体素大小是点云下采样大小设置，单位为m。
4. 输出的点云图为基于高度的热力图，红色表示位置更高。
5. 软件启动和配准过程都可能比较慢，请耐心等待。

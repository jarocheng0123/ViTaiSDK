import time
import cv2
import numpy as np
import asyncio
import pyautogui  # 用于模拟键盘鼠标操作
from pyvitaisdk import GF225, VTSDeviceFinder
from datetime import datetime

# https://www.4399.com/flash/93398_3.htm

# ================ 参数配置区 ================
# 防抖与过滤参数
DEBOUNCE_THRESHOLD = 0.1          # 防抖距离阈值（调小更灵敏）
SAMPLE_FPS = 1                    # 采样帧率

# 方向检测参数
HORIZONTAL_THRESHOLD = 0.4           # 左右灵敏度
HORIZONTAL_WEIGHT = 0.3             # 前后方向
Z_THRESHOLD_1 = 50.0               # 下1阈值（浅按压）
Z_THRESHOLD_2 = 70.0               # 下2阈值（中按压）
Z_THRESHOLD_3 = 100.0              # 下3阈值（深按压）

# 新增：按键超时与无操作超时参数
MAX_KEY_PRESS_DURATION = 5.0  # 按键最长持续时间（秒）
IDLE_TIMEOUT = 10.0           # 无操作超时时间（秒）

# 彩色输出配置
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET = "\033[0m"

# 按键映射配置
KEY_MAPPING = {
    # "前": "w",
    # "后": "s",
    # "左": "a",
    # "右": "d",
    "前": "up",      # 上方向键
    "后": "down",    # 下方向键
    "左": "left",    # 左方向键
    "右": "right",   # 右方向键
    "下1": "space",
    "下2": "space",
    "下3": "space",
}

# 鼠标映射配置
MOUSE_MAPPING = {
    # "下1": "left",
    # "下2": "right",
    # "下3": "middle"
}

# "left" 对应鼠标左键
# "right" 对应鼠标右键
# "middle" 对应鼠标中键

# ================ 程序逻辑区 ================
class GF225Manager:
    """传感器设备管理层"""

    def __init__(self):
        self._gf225 = None
        self.xyz_data = None
        self.origin_xyz_list = None
        self.current_xyz_list = None
        self.task = None
        self._is_first = True
        self._fps = SAMPLE_FPS
        self._elapsed = 1 / self._fps
        self.debounce_threshold = DEBOUNCE_THRESHOLD


    async def connect_device(self):
        """连接GF225硬件设备"""
        if self._gf225 is None:
            finder = VTSDeviceFinder()
            if not finder.get_sns():
                print("未找到GF225设备")
                raise RuntimeError("未找到GF225设备")

            config = finder.get_device_by_sn(finder.get_sns()[0])
            self._gf225 = GF225(
                config=config,
                marker_size=20
            )
            self._gf225.start_backend()
            self._gf225.calibrate(10)
            self._gf225.set_warp_params(mode='auto')


    async def debounce(self, list1, list2):
        """防抖处理：过滤微小抖动"""
        arr1 = np.array(list1)
        arr2 = np.array(list2)

        ox, oy = arr1[:, 0], arr1[:, 1]
        cx, cy = arr2[:, 0], arr2[:, 1]

        dx = cx - ox
        dy = cy - oy

        dist_squared = dx **2 + dy** 2
        K = np.where(dist_squared < self.debounce_threshold **2, 0, 1)

        arr2[:, 0] = ox + K * dx
        arr2[:, 1] = oy + K * dy

        return arr2.tolist(), K * dx, K * dy


    async def data_generator(self):
        """数据生成协程"""
        while True:
            t1 = time.time()
            
            warped_frame = self._gf225.get_warped_frame()

            if self._gf225.is_calibrate():
                self._gf225.tracking(warped_frame)
                self._gf225.recon3d(warped_frame)
                depth_map = self._gf225.get_depth_map()
                depth_frame = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                current_markers = self._gf225.get_markers()
                current_markers_int = np.squeeze(current_markers.astype(np.int16).reshape((1, -1, 2)))
                current_markers_int = np.clip(current_markers_int, a_min=0, a_max=depth_frame.shape[0] - 1)
                
                rows, columns = current_markers_int[:, 1], current_markers_int[:, 0]
                z = depth_frame[rows, columns]
                xyz = np.column_stack((current_markers_int, z))
                xyz_list = xyz.tolist()

                if self._is_first:
                    origin_markers = self._gf225.get_origin_markers()
                    origin_markers_int = np.squeeze(origin_markers.astype(np.int16).reshape((1, -1, 2)))
                    origin_markers_int = np.clip(origin_markers_int, a_min=0, a_max=depth_frame.shape[0] - 1)
                    rows_origin, cols_origin = origin_markers_int[:, 1], origin_markers_int[:, 0]
                    z_origin = depth_frame[rows_origin, cols_origin]
                    origin_xyz = np.column_stack((origin_markers_int, z_origin))
                    self.origin_xyz_list = origin_xyz.tolist()
                    self._is_first = False

                debounced_xyz_list, dx, dy = await self.debounce(self.origin_xyz_list, xyz_list)
                self.current_xyz_list = debounced_xyz_list

                if len(dx) > 0 and len(dy) > 0 and len(z) > 0:
                    mean_x = np.mean(dx)
                    mean_y = np.mean(dy)
                    mean_z = np.mean(z)

                    self.xyz_data = {
                        "x": round(float(mean_x), 6),
                        "y": round(float(mean_y), 6),
                        "z": round(float(mean_z), 6)
                    }

            elapsed = time.time() - t1
            if elapsed < self._elapsed:
                await asyncio.sleep(self._elapsed - elapsed)


def detect_direction(current_xyz, previous_xyz=None):
    """增强左右方向的识别灵敏度，区分下1/下2/下3"""
    if previous_xyz is None:
        return "初始化中..."
    
    dx = current_xyz["x"] - previous_xyz["x"]
    dy = current_xyz["y"] - previous_xyz["y"]
    dz = current_xyz["z"] - previous_xyz["z"]
    
    # 1. 优先检测左右方向
    if abs(dx) > HORIZONTAL_THRESHOLD:
        return "左" if dx > 0 else "右"
    
    # 2. 检测前后方向
    elif abs(dy) > HORIZONTAL_WEIGHT:
        return "前" if dy > 0 else "后"
    
    # 3. 检测按压方向（区分下1/下2/下3）
    elif dz > Z_THRESHOLD_3:
        return "下3"
    elif dz > Z_THRESHOLD_2:
        return "下2"
    elif dz > Z_THRESHOLD_1:
        return "下1"
    
    return "无操作"


async def main():
    # 初始化pyautogui
    pyautogui.PAUSE = 0.05
    pyautogui.FAILSAFE = False

    manager = GF225Manager()
    previous_xyz = None
    current_key = None  # 记录当前按下的键
    mouse_pressed_button = None  # 记录当前按下的鼠标按键
    
    # 按键计时与无操作计时
    key_press_start_time = 0
    last_operation_time = time.time()

    def format_value(value):
        return f"{value:.6f}" if value is not None else "------"
    
    try:
        await manager.connect_device()
        print("设备连接成功，开始采集数据...")
        print(f"操作映射：前→W，后→S，左→A，右→D，下1→J，下2→K，下3→L")
        print(f"按键超时：{MAX_KEY_PRESS_DURATION}秒，无操作超时：{IDLE_TIMEOUT}秒")
        manager.task = asyncio.create_task(manager.data_generator())
        
        while True:
            if manager.xyz_data:
                current_xyz = manager.xyz_data
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 计算偏移量
                offset_from_previous = (
                    current_xyz["x"] - previous_xyz["x"] if previous_xyz else None,
                    current_xyz["y"] - previous_xyz["y"] if previous_xyz else None,
                    current_xyz["z"] - previous_xyz["z"] if previous_xyz else None
                )
                
                # 检测方向
                direction = detect_direction(current_xyz, previous_xyz)
                
                # 更新最后操作时间
                if direction != "无操作":
                    last_operation_time = time.time()
                
                # 彩色输出
                if direction in ["前", "后"]:
                    colored_dir = f"{COLOR_RED}{direction}{COLOR_RESET}"
                elif direction in ["左", "右"]:
                    colored_dir = f"{COLOR_GREEN}{direction}{COLOR_RESET}"
                elif direction in ["下1", "下2", "下3"]:
                    colored_dir = f"{COLOR_YELLOW}{direction}{COLOR_RESET}"
                else:
                    colored_dir = direction

                # 打印信息
                print(
                    f"{current_time} - "
                    f"当前坐标：X={current_xyz['x']:.6f}, Y={current_xyz['y']:.6f}, Z={current_xyz['z']:.6f}\n"
                    f"  相对偏移：dx={format_value(offset_from_previous[0])}, dy={format_value(offset_from_previous[1])}, dz={format_value(offset_from_previous[2])}\n"
                    f"  操作方向: {colored_dir}\n"
                )

                # 检查按键超时
                if current_key and (time.time() - key_press_start_time > MAX_KEY_PRESS_DURATION):
                    print(f"按键超时：{current_key}持续时间超过{MAX_KEY_PRESS_DURATION}秒，已释放")
                    pyautogui.keyUp(KEY_MAPPING[current_key])
                    current_key = None
                
                # 检查鼠标超时
                if mouse_pressed_button and (time.time() - key_press_start_time > MAX_KEY_PRESS_DURATION):
                    print(f"鼠标超时：{mouse_pressed_button}持续时间超过{MAX_KEY_PRESS_DURATION}秒，已释放")
                    pyautogui.mouseUp(button=mouse_pressed_button)
                    mouse_pressed_button = None
                
                # 检查无操作超时
                if time.time() - last_operation_time > IDLE_TIMEOUT:
                    print(f"无操作超时：超过{IDLE_TIMEOUT}秒未检测到有效操作，已释放所有按键和鼠标")
                    if current_key:
                        pyautogui.keyUp(KEY_MAPPING[current_key])
                        current_key = None
                    if mouse_pressed_button:
                        pyautogui.mouseUp(button=mouse_pressed_button)
                        mouse_pressed_button = None
                
                # 键盘映射逻辑
                if direction in KEY_MAPPING:
                    target_key = KEY_MAPPING[direction]
                    if target_key != current_key:
                        # 释放当前按键（如果不同）
                        if current_key:
                            pyautogui.keyUp(KEY_MAPPING[current_key])
                        # 释放当前鼠标按键
                        if mouse_pressed_button:
                            pyautogui.mouseUp(button=mouse_pressed_button)
                            mouse_pressed_button = None
                        # 按下新按键
                        pyautogui.keyDown(target_key)
                        current_key = direction
                        key_press_start_time = time.time()  # 记录开始时间
                
                # 鼠标映射逻辑
                elif direction in MOUSE_MAPPING:
                    target_button = MOUSE_MAPPING[direction]
                    if target_button != mouse_pressed_button:
                        # 释放当前按键
                        if current_key:
                            pyautogui.keyUp(KEY_MAPPING[current_key])
                            current_key = None
                        # 释放当前鼠标按键（如果不同）
                        if mouse_pressed_button and mouse_pressed_button != target_button:
                            pyautogui.mouseUp(button=mouse_pressed_button)
                        # 按下新鼠标按键
                        pyautogui.mouseDown(button=target_button)
                        mouse_pressed_button = target_button
                        key_press_start_time = time.time()  # 记录开始时间
                
                # 无有效操作时释放所有按键和鼠标
                elif current_key or mouse_pressed_button:
                    if current_key:
                        pyautogui.keyUp(KEY_MAPPING[current_key])
                        current_key = None
                    if mouse_pressed_button:
                        pyautogui.mouseUp(button=mouse_pressed_button)
                        mouse_pressed_button = None

                # 更新上一时刻坐标
                previous_xyz = current_xyz
            
            await asyncio.sleep(1 / manager._fps)
    
    except KeyboardInterrupt:
        print("用户终止程序")
    except Exception as e:
        print(f"运行错误：{str(e)}")
    finally:
        # 强制释放所有按键和鼠标
        for key in KEY_MAPPING.values():
            pyautogui.keyUp(key)
        for button in MOUSE_MAPPING.values():
            pyautogui.mouseUp(button=button)
        print("所有按键和鼠标已强制释放")
        
        # 清理设备资源
        if manager.task:
            try:
                manager.task.cancel()
                await manager.task
            except asyncio.CancelledError:
                print("数据生成任务已取消")
        if manager._gf225:
            try:
                manager._gf225.stop_backend()
                manager._gf225.release()
                print("设备资源已释放")
            except Exception as e:
                print(f"释放设备资源时出错：{str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
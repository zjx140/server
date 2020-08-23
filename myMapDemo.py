import os
import cv2
from selenium import webdriver
from myCommonModules import showImg

# 定义地图类，演示用
# mapPath=None,地图路径
# mapType='BMAP_NORMAL_MAP',地图类型
# chromeDriverPath=None,ChromeDriver路径
# sensitivity=1, 手势响应灵敏度
class baiduMap:
    def __init__(self,mapPath=None,mapType='BMAP_NORMAL_MAP',chromeDriverPath=None,sensitivity=1):
        self.canDemo=True # 地图是否可以演示
        self.url=''
        self.driver=None
        self.mapType=mapType # 地图类型'BMAP_NORMAL_MAP'或'BMAP_SATELLITE_MAP'
        # BMAP_NORMAL_MAP 此地图类型展示普通街道视图
        # BMAP_SATELLITE_MAP 此地图类型展示卫星视图
        # BMAP_HYBRID_MAP 此地图类型展示卫星和路网的混合视图

        # 脚本动作，key对应手势名称，value对应脚本动作
        self.jsAction = {'moveleft': 'map.panBy(-50, 0)',
                        'moveright': 'map.panBy(50, 0)',
                        'moveup': 'map.panBy(0, -50)',
                        'movedown': 'map.panBy(0, 50)',
                        'zoomin': 'map.zoomIn()',
                        'zoomout': 'map.zoomOut()',
                        'turnleft': 'map.setMapType(BMAP_NORMAL_MAP)',
                        'turnright': 'map.setMapType(BMAP_SATELLITE_MAP)'}

        self.lastGesture='invalid'# 记录上一次手势，防止重复动作闪屏
        self.lastGestureTimes = 0# 记录上一次手势重复次数
        self.waitingforClose = False# 是否等待退出，确认就退出，取消才继续相应其他操作
        self.confirmClose=False# 确认关闭
        self.sensitivity=sensitivity # 灵敏度

        if mapPath is None:
            self.canDemo = False
        else:
            self.url = mapPath
        # print(url)
        if chromeDriverPath is None:
            print('未设定Web浏览器，将自动关闭人机交互演示程序。')
            self.canDemo = False
        elif os.path.exists(chromeDriverPath):
            print('启动Web浏览器...')
            self.driver = webdriver.Chrome(chromeDriverPath)
            self.driver.get(self.url)
        else:
            print('未找到', chromeDriverPath, '无法启动Web浏览器，将自动关闭人机交互演示程序。')
            self.canDemo = False
    def run(self,command):
        if command=='close':
            print('收到close手势，等待确认关闭...')
            self.waitingforClose=True #除了确认和取消，不响应其他手势
            dialogfilename='..\mapDemo\dialog.jpg'
            if os.path.exists(dialogfilename):
                ImgDialog=cv2.imread(dialogfilename)
                showImg(ImgDialog,'Close?')
        if self.waitingforClose:
            if command=='ok':
                print('收到ok手势，确认关闭')
                self.confirmClose=True
            elif command=='cancel':
                print('收到cancel手势，取消关闭')
                cv2.destroyWindow('Close?')
                self.waitingforClose=False
        if self.jsAction.get(command) and not self.waitingforClose:  # 执行手势对应的动作脚本
            if self.lastGesture == command:#记录同一手势重复次数
                self.lastGestureTimes +=1
            else:
                self.lastGestureTimes=0
            print('收到可响应的手势：', command,'此手势已连续重复次数：',self.lastGestureTimes)
            if not ('turn' in command and self.lastGesture == command):  # 避免连续翻页
                if not ((self.mapType == 'BMAP_NORMAL_MAP' and 'BMAP_NORMAL_MAP' in self.jsAction[command]) or (
                        self.mapType == 'BMAP_SATELLITE_MAP' and 'BMAP_SATELLITE_MAP' in self.jsAction[command])):
                    if 'BMAP_NORMAL_MAP' in self.jsAction[command]:
                        self.mapType = 'BMAP_NORMAL_MAP'
                    if 'BMAP_SATELLITE_MAP' in self.jsAction[command]:
                        self.mapType = 'BMAP_SATELLITE_MAP'
                    if 'zoom' in command: # 缩放操作用灵敏度调一下
                        # 第一次，或每间隔设定灵敏度值才执行一次
                        if (self.lastGestureTimes==0 or (self.lastGestureTimes % self.sensitivity==0)):
                            self.driver.execute_script(self.jsAction[command])
                            print('执行动作：', self.jsAction[command])
                    else:
                        self.driver.execute_script(self.jsAction[command])
                        print('执行动作：', self.jsAction[command])
            self.lastGesture = command

# 基于yolov11的占道经营街道摆摊检测系统python源码+pytorch模型+评估指标曲线+精美GUI界面

【算法介绍】

在城市街道管理朝着规范化、智能化方向持续推进的重要阶段，精准且高效地监测街道上的占道经营情况，已然成为维护城市秩序、提升城市形象以及保障市民出行便利的核心挑战之一。街道环境复杂多样，摊位分布灵活多变，像不同时间段的人流量差异导致的摊位聚集位置变化、摊位所售商品种类繁多带来的形态各异以及摊位所处的不同街道位置（如路口、人行道边缘等）等，这些状况不仅直观反映了街道当下的管理秩序，更与城市的整体运行效率、居民的生活质量等紧密相连。一旦出现占道经营异常情况，如部分摊位严重超出规定经营区域、在禁止摆摊时段违规经营或者因摊位摆放杂乱导致交通堵塞等，若未能及时察觉并采取相应措施，极易引发交通混乱、影响周边商铺正常营业甚至引发安全事故，给城市管理带来巨大的压力和挑战。

传统街道占道经营检测方式主要依赖人工巡查。然而，受城市街道范围不断扩大、摊位数量日益增多以及街道地形复杂等因素限制，人工巡查很难全面覆盖街道的各个角落，尤其是那些隐藏在小巷深处或处于偏远位置的摊位情况，往往难以被及时观察到。而且，早期基于简单人工判断的监测方法，由于摊位经营形式多样、不同时段人流量和车流量变化大以及各种建筑物、树木遮挡等因素干扰，误判率高达 40%以上，根本无法满足城市管理者“精准化、全方位”的管理需求。因此，开发一套具备高精度、强适应性且能实时监测的街道占道经营智能检测系统，成为提升城市管理水平和治理效能的关键技术突破点。

目前现有技术存在诸多明显瓶颈：人工巡查不仅效率极其低下（单人单日仅能完成有限街道长度的巡查），而且巡查人员还面临着被车辆碰撞、恶劣天气影响等风险；基于简单图像特征提取的传统算法，难以准确区分正常经营摊位与违规占道经营摊位（例如，部分摊位在规定经营区域边缘的微小越界情况），在光线昏暗、天气恶劣（如暴雨、雾霾）等低能见度环境下，算法性能会急剧下降；传统目标检测模型对摊位经营的多变性（如不同经营时段摊位的规模变化、不同商品种类的摆放差异）和尺度变化（从小型手推车摊位到大型帐篷摊位）适应性较差，对于小目标（如单个售卖小物件的摊位）的漏检率超过 50%，难以满足实际城市街道管理的复杂需求。

基于 YOLOv11 的占道经营街道摆摊检测系统为城市街道管理带来了革命性的变革。YOLOv11 作为先进的目标检测算法，具备强大的特征提取和实时检测能力。该系统充分发挥 YOLOv11 的端到端实时检测优势，并针对街道复杂环境进行了深度优化。

此系统能够精准识别“zdjy（占道经营）”这一类别。它可以准确捕捉占道经营摊位的各种形态特征，无论是摊位部分超出经营区域边界，还是完全堵塞交通要道；无论是售卖水果、小吃的小型摊位，还是售卖衣物、日用品的大型摊位，系统都能敏锐识别。通过对大量街道占道经营图像数据的学习和训练，系统能够保持较高的检测准确率，即便在光线变化大、人流量密集、存在各种遮挡物等复杂场景下，也能稳定发挥检测性能。

同时，系统具备强大的抗干扰能力，能够有效应对街道上的车辆、行人、建筑物等干扰因素，不会因这些干扰而出现误判或漏判的情况。这为新型智能化城市街道管理建设提供了坚实的技术支撑，助力城市管理者实现高效、精准、科学的街道管理，及时发现并处理占道经营问题，维护城市街道的正常秩序和良好形象。

【效果展示】

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/74efac96d24440fea0f0c702b31f5854.jpeg"></div>

<div style="text-align:center;"><img alt="" src="https://i-blog.csdnimg.cn/direct/7154d94ad2b448a19f78d69e04992b7f.jpeg"></div>

【测试环境】

windows10
anaconda3+python3.8
torch==2.3.1
ultralytics==8.3.81

【模型可以检测出类别】

zdjy(占道经营拼音缩写)

【训练数据集介绍】

数据集格式：Pascal VOC格式+YOLO格式(不包含分割路径的txt文件，仅仅包含jpg图片以及对应的VOC格式xml文件和yolo格式txt文件)
图片数量(jpg文件个数)：5226
标注数量(xml文件个数)：5226
标注数量(txt文件个数)：5226
标注类别数：1
标注类别名称:["zdjy"]

所在仓库：firc-dataset
每个类别标注的框数：
zdjy 框数 = 5799
总框数：5799
使用标注工具：labelImg
标注规则：对类别进行画矩形框
重要说明：暂无
特别声明：本数据集不对训练的模型或者权重文件精度作任何保证，数据集只提供准确且合理标注

图片预览：

 ![](./assets/334_3.jpeg)

 ![](./assets/334_4.jpeg)

标注例子：

 ![](./assets/334_5.jpeg)

【训练信息】

| 参数 | 值 |
|:---:|:---:|
| 训练集图片数 | 4964 |
| 验证集图片数 | 262 |
| 训练map | 96.8% |
| 训练精度(Precision) | 94.0% |
| 训练召回率(Recall) | 92.3% |

【界面设计】

```
class Ui_MainWindow(QtWidgets.QMainWindow):
    signal = QtCore.pyqtSignal(str, str)
 
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1280, 728)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
 
        self.weights_dir = './weights'
 
        self.picture = QtWidgets.QLabel(self.centralwidget)
        self.picture.setGeometry(QtCore.QRect(260, 10, 1010, 630))
        self.picture.setStyleSheet("background:black")
        self.picture.setObjectName("picture")
        self.picture.setScaledContents(True)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 81, 21))
        self.label_2.setObjectName("label_2")
        self.cb_weights = QtWidgets.QComboBox(self.centralwidget)
        self.cb_weights.setGeometry(QtCore.QRect(10, 40, 241, 21))
        self.cb_weights.setObjectName("cb_weights")
        self.cb_weights.currentIndexChanged.connect(self.cb_weights_changed)
 
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 70, 72, 21))
        self.label_3.setObjectName("label_3")
        self.hs_conf = QtWidgets.QSlider(self.centralwidget)
        self.hs_conf.setGeometry(QtCore.QRect(10, 100, 181, 22))
        self.hs_conf.setProperty("value", 25)
        self.hs_conf.setOrientation(QtCore.Qt.Horizontal)
        self.hs_conf.setObjectName("hs_conf")
        self.hs_conf.valueChanged.connect(self.conf_change)
        self.dsb_conf = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_conf.setGeometry(QtCore.QRect(200, 100, 51, 22))
        self.dsb_conf.setMaximum(1.0)
        self.dsb_conf.setSingleStep(0.01)
        self.dsb_conf.setProperty("value", 0.25)
        self.dsb_conf.setObjectName("dsb_conf")
        self.dsb_conf.valueChanged.connect(self.dsb_conf_change)
        self.dsb_iou = QtWidgets.QDoubleSpinBox(self.centralwidget)
        self.dsb_iou.setGeometry(QtCore.QRect(200, 160, 51, 22))
        self.dsb_iou.setMaximum(1.0)
        self.dsb_iou.setSingleStep(0.01)
        self.dsb_iou.setProperty("value", 0.45)
        self.dsb_iou.setObjectName("dsb_iou")
        self.dsb_iou.valueChanged.connect(self.dsb_iou_change)
        self.hs_iou = QtWidgets.QSlider(self.centralwidget)
        self.hs_iou.setGeometry(QtCore.QRect(10, 160, 181, 22))
        self.hs_iou.setProperty("value", 45)
        self.hs_iou.setOrientation(QtCore.Qt.Horizontal)
        self.hs_iou.setObjectName("hs_iou")
        self.hs_iou.valueChanged.connect(self.iou_change)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 130, 72, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(10, 210, 72, 21))
        self.label_5.setObjectName("label_5")
        self.le_res = QtWidgets.QTextEdit(self.centralwidget)
        self.le_res.setGeometry(QtCore.QRect(10, 240, 241, 400))
        self.le_res.setObjectName("le_res")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1110, 30))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(self)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionopenpic = QtWidgets.QAction(self)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionopenpic.setIcon(icon)
        self.actionopenpic.setObjectName("actionopenpic")
        self.actionopenpic.triggered.connect(self.open_image)
        self.action = QtWidgets.QAction(self)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action.setIcon(icon1)
        self.action.setObjectName("action")
        self.action.triggered.connect(self.open_video)
        self.action_2 = QtWidgets.QAction(self)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/3.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_2.setIcon(icon2)
        self.action_2.setObjectName("action_2")
        self.action_2.triggered.connect(self.open_camera)
 
        self.actionexit = QtWidgets.QAction(self)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/4.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionexit.setIcon(icon3)
        self.actionexit.setObjectName("actionexit")
        self.actionexit.triggered.connect(self.exit)
 
        self.toolBar.addAction(self.actionopenpic)
        self.toolBar.addAction(self.action)
        self.toolBar.addAction(self.action_2)
        self.toolBar.addAction(self.actionexit)
 
        self.retranslateUi()
        QtCore.QMetaObject.connectSlotsByName(self)
        self.init_all()
```



【常用评估参数介绍】

在目标检测任务中，评估模型的性能是至关重要的。你提到的几个术语是评估模型性能的常用指标。下面是对这些术语的详细解释：

Class：
这通常指的是模型被设计用来检测的目标类别。例如，一个模型可能被训练来检测车辆、行人或动物等不同类别的对象。
Images：
表示验证集中的图片数量。验证集是用来评估模型性能的数据集，与训练集分开，以确保评估结果的公正性。
Instances：
在所有图片中目标对象的总数。这包括了所有类别对象的总和，例如，如果验证集包含100张图片，每张图片平均有5个目标对象，则Instances为500。
P（精确度Precision）：
精确度是模型预测为正样本的实例中，真正为正样本的比例。计算公式为：Precision = TP / (TP + FP)，其中TP表示真正例（True Positives），FP表示假正例（False Positives）。
R（召回率Recall）：
召回率是所有真正的正样本中被模型正确预测为正样本的比例。计算公式为：Recall = TP / (TP + FN)，其中FN表示假负例（False Negatives）。
mAP50：
表示在IoU（交并比）阈值为0.5时的平均精度（mean Average Precision）。IoU是衡量预测框和真实框重叠程度的指标。mAP是一个综合指标，考虑了精确度和召回率，用于评估模型在不同召回率水平上的性能。在IoU=0.5时，如果预测框与真实框的重叠程度达到或超过50%，则认为该预测是正确的。
mAP50-95：
表示在IoU从0.5到0.95（间隔0.05）的范围内，模型的平均精度。这是一个更严格的评估标准，要求预测框与真实框的重叠程度更高。在目标检测任务中，更高的IoU阈值意味着模型需要更准确地定位目标对象。mAP50-95的计算考虑了从宽松到严格的多个IoU阈值，因此能够更全面地评估模型的性能。
这些指标共同构成了评估目标检测模型性能的重要框架。通过比较不同模型在这些指标上的表现，可以判断哪个模型在实际应用中可能更有效。

【使用步骤】

使用步骤：
（1）首先根据官方框架ultralytics安装教程安装好yolov11环境，并安装好pyqt5
（2）切换到自己安装的yolo11环境后，并切换到源码目录，执行python main.py即可运行启动界面，进行相应的操作即可

【提供文件】

python源码
yolo11n.pt模型
训练的map,P,R曲线图(在weights\results.png)
测试图片（在test_img文件夹下面）

注意提供训练的数据集，请到mytxt.txt文件中找到地址
<br>项目源码地址：https://mbd.pub/o/bread/YZWVk5lwZQ==

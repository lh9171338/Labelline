import sys
import os
import numpy as np
import glob
import cv2
import logging
import scipy.io as sio
from yacs.config import CfgNode
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import camera as cam


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()

        # Variables
        logging.basicConfig(level=logging.DEBUG)
        self.image_folder = cfg.image_folder
        self.label_folder = cfg.label_folder
        self.file_list = []
        self.file_index = 0
        self.num_file = 0
        self.image = None
        self.auxiliary_image = None
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.line_index = len(self.lines) - 1
        self.label_endpoint = None
        self.capture_endpoint = None

        self.save_flag = True
        self.transform_flag = False
        self.move_flag = 0
        self.label_flag = False

        self.camera = cam.Pinhole()
        self.decimal_precision = cfg.decimal_precision
        self.default_image_size = cfg.default_image_size

        # Parameters
        self.scale = cfg.init_scale
        self.scale_limit = cfg.scale_limit
        self.line_width = cfg.line_width
        self.point_radius = cfg.point_radius
        self.point_select_thresh = 2 * self.point_radius if cfg.point_select_thresh is None else cfg.point_select_thresh
        self.line_select_thresh = cfg.line_select_thresh
        self.point_align_thresh = cfg.point_align_thresh
        self.patterns = cfg.patterns

        # UI
        self.menuBar = QMenuBar()
        self.menu_File = self.menuBar.addMenu('File')
        self.menu_Edit = self.menuBar.addMenu('Edit')
        self.menu_Help = self.menuBar.addMenu('Help')
        self.menu_OpenDir = self.menu_File.addAction(QIcon('icon/open.png'), 'OpenDir')
        self.menu_File.addSeparator()
        self.menu_Save = self.menu_File.addAction(QIcon('icon/save.png'), 'Save')
        self.menu_Next = self.menu_Edit.addAction(QIcon('icon/next.png'), 'Next')
        self.menu_Edit.addSeparator()
        self.menu_Prev = self.menu_Edit.addAction(QIcon('icon/prev.png'), 'Prev')
        self.menu_Edit.addSeparator()
        self.menu_Create = self.menu_Edit.addAction(QIcon('icon/create.png'), 'Create')
        self.menu_Edit.addSeparator()
        self.menu_Delete = self.menu_Edit.addAction(QIcon('icon/delete.png'), 'Delete')
        self.menu_Edit.addSeparator()
        self.menu_Transform = self.menu_Edit.addAction(QIcon('icon/transform.png'), 'Transform')
        self.menu_Edit.addSeparator()
        self.menu_Exchange = self.menu_Edit.addAction(QIcon('icon/exchange.png'), 'Exchange')
        self.menu_Tutorial = self.menu_Help.addAction(QIcon('icon/tutorial.png'), 'Tutorial')

        self.button_OpenDir = QPushButton(text='Open Dir', icon=QIcon('icon/open.png'))
        self.button_Save = QPushButton(text='Save', icon=QIcon('icon/save.png'))
        self.button_Next = QPushButton(text='Next', icon=QIcon('icon/next.png'))
        self.button_Prev = QPushButton(text='Prev', icon=QIcon('icon/prev.png'))
        self.button_Create = QPushButton(text='Create', icon=QIcon('icon/create.png'))
        self.button_Delete = QPushButton(text='Delete', icon=QIcon('icon/delete.png'))
        self.button_ZoomIn = QPushButton(text='Zoom In', icon=QIcon('icon/zoom-in.png'))
        self.button_ZoomOut = QPushButton(text='Zoom Out', icon=QIcon('icon/zoom-out.png'))
        self.text_zoom = QLineEdit(text='100%', alignment=Qt.AlignCenter)
        self.text_zoom.setValidator(QRegExpValidator(QRegExp('\d+%?')))
        self.text_zoom.setFixedSize(self.button_OpenDir.sizeHint())
        self.buttonLayout = QVBoxLayout()
        self.label_Image = QLabel()
        self.label_AuxiliaryImage = QLabel()
        self.imageLayout = QHBoxLayout()

        self.text_Line = QLabel('Line list')
        self.text_File = QLabel('File list')
        self.list_Line = QListWidget()
        self.list_File = QListWidget()
        self.listLayout = QVBoxLayout()

        self.layout = QHBoxLayout()
        self.centralWidget = QWidget()

        self.InitUI(cfg)

    def InitUI(self, cfg):
        # Set UI
        self.buttonLayout.addWidget(self.button_OpenDir)
        self.buttonLayout.addWidget(self.button_Save)
        self.buttonLayout.addWidget(QFrame(frameShape=QFrame.HLine))
        self.buttonLayout.addWidget(self.button_Next)
        self.buttonLayout.addWidget(self.button_Prev)
        self.buttonLayout.addWidget(self.button_Create)
        self.buttonLayout.addWidget(self.button_Delete)
        self.buttonLayout.addWidget(QFrame(frameShape=QFrame.HLine))
        self.buttonLayout.addWidget(self.text_zoom)
        self.buttonLayout.addWidget(self.button_ZoomIn)
        self.buttonLayout.addWidget(self.button_ZoomOut)
        self.imageLayout.addWidget(self.label_Image, Qt.AlignCenter)
        self.imageLayout.addWidget(self.label_AuxiliaryImage, Qt.AlignCenter)
        self.listLayout.addWidget(self.text_Line)
        self.listLayout.addWidget(self.list_Line)
        self.listLayout.addWidget(self.text_File)
        self.listLayout.addWidget(self.list_File)
        self.layout.addLayout(self.buttonLayout)
        self.layout.addStretch(1)
        self.layout.addLayout(self.imageLayout)
        self.layout.addStretch(1)
        self.layout.addLayout(self.listLayout)
        self.centralWidget.setLayout(self.layout)
        self.setMenuBar(self.menuBar)
        self.setCentralWidget(self.centralWidget)
        self.setWindowTitle('Labelline')
        self.resize(cfg.window_size[0], cfg.window_size[1])

        # Register callback
        self.button_OpenDir.clicked.connect(self.OpenDir_Callback)
        self.button_Save.clicked.connect(self.Save_Callback)
        self.button_Next.clicked.connect(self.Next_Callback)
        self.button_Prev.clicked.connect(self.Prev_Callback)
        self.button_Create.clicked.connect(self.Create_Callback)
        self.button_Delete.clicked.connect(self.Delete_Callback)
        self.button_ZoomIn.clicked.connect(self.ZoomIn_Callback)
        self.button_ZoomOut.clicked.connect(self.ZoomOut_Callback)

        self.menu_OpenDir.triggered.connect(self.OpenDir_Callback)
        self.menu_Save.triggered.connect(self.Save_Callback)
        self.menu_Next.triggered.connect(self.Next_Callback)
        self.menu_Prev.triggered.connect(self.Prev_Callback)
        self.menu_Create.triggered.connect(self.Create_Callback)
        self.menu_Delete.triggered.connect(self.Delete_Callback)
        self.menu_Transform.triggered.connect(self.Transform_Callback)
        self.menu_Exchange.triggered.connect(self.Exchange_Callback)
        self.menu_Tutorial.triggered.connect(self.Tutorial_Callback)

        self.menu_OpenDir.setShortcut(cfg.menu_OpenDir_shortcut)
        self.menu_Save.setShortcut(cfg.menu_Save_shortcut)
        self.menu_Next.setShortcut(cfg.menu_Next_shortcut)
        self.menu_Prev.setShortcut(cfg.menu_Prev_shortcut)
        self.menu_Create.setShortcut(cfg.menu_Create_shortcut)
        self.menu_Delete.setShortcut(cfg.menu_Delete_shortcut)
        self.menu_Transform.setShortcut(cfg.menu_Transform_shortcut)
        self.menu_Exchange.setShortcut(cfg.menu_Exchange_shortcut)
        self.menu_Tutorial.setShortcut(cfg.menu_Tutorial_shortcut)

        self.list_Line.clicked.connect(self.ListLine_Callback)
        self.list_File.clicked.connect(self.ListFile_Callback)
        self.text_zoom.editingFinished.connect(self.TextZoom_Callback)

        self.label_Image.mouseReleaseEvent = self.mouseRelease_Callback
        self.label_Image.mouseMoveEvent = self.mouseMove_Callback
        self.label_Image.setMouseTracking(True)

        self.reset()
        self.button_OpenDir.setEnabled(True)
        self.menu_OpenDir.setEnabled(True)

    def reset(self):
        self.list_Line.setEnabled(False)
        self.list_File.setEnabled(False)
        self.text_zoom.setEnabled(False)

        self.button_OpenDir.setEnabled(False)
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(False)
        self.button_Prev.setEnabled(False)
        self.button_Create.setEnabled(False)
        self.button_Delete.setEnabled(False)
        self.button_ZoomIn.setEnabled(False)
        self.button_ZoomOut.setEnabled(False)

        self.menu_OpenDir.setEnabled(False)
        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(False)
        self.menu_Prev.setEnabled(False)
        self.menu_Create.setEnabled(False)
        self.menu_Delete.setEnabled(False)
        self.menu_Transform.setEnabled(False)
        self.menu_Exchange.setEnabled(False)

        self.label_Image.setEnabled(False)

    def data_update(self):
        self.label_flag = False
        self.label_endpoint = None
        self.capture_endpoint = None

        image_file = self.file_list[self.file_index]
        filename = os.path.basename(image_file)
        filename = os.path.splitext(filename)[0] + '.mat'
        line_file = os.path.join(self.data_path, self.label_folder, filename)

        image = cv2.imread(image_file)
        if image is None:
            logging.error('Image must not be None')
            exit()
        width, height = image.shape[1] // 2, image.shape[0]
        self.image = image[:, :width]
        self.auxiliary_image = image[:, width:]

        self.lines = np.zeros((0, 2, 2), np.float32)
        if os.path.isfile(line_file):
            lines = sio.loadmat(line_file)['lines']
            if len(lines):
                # 去除长度为0的线段
                mask = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=-1) > 0
                lines = lines[mask]

                # 合并相同的端点
                juncs = np.unique(np.vstack((lines[:, 1], lines[:, 0])), axis=0)
                jids = {}
                for id, junc in enumerate(juncs):
                    jids[tuple(junc)] = id
                dists = np.linalg.norm(juncs[:, None] - juncs[None], axis=-1)
                indices = np.argwhere(dists < self.point_select_thresh)
                mask = indices[:, 0] < indices[:, 1]
                indices = indices[mask]
                if len(indices):
                    print(os.path.basename(line_file))
                    for i, j in indices:
                        junc_i = juncs[i]
                        junc_j = juncs[j]
                        jids[tuple(junc_j)] = jids[tuple(junc_i)]
                for i in range(len(lines)):
                    id1 = jids[tuple(lines[i, 0])]
                    id2 = jids[tuple(lines[i, 1])]
                    lines[i, 0] = juncs[id1]
                    lines[i, 1] = juncs[id2]

                # 限制线段端点范围
                lines[:, :, 0] = np.clip(lines[:, :, 0], 0, width - 1e-4)
                lines[:, :, 1] = np.clip(lines[:, :, 1], 0, height - 1e-4)

                self.lines = lines
                self.Save_Callback()
            else:
                os.remove(line_file)
        self.line_index = len(self.lines) - 1

    def transform(self):
        image = self.image.copy()
        auxiliary_image = self.auxiliary_image.copy()
        if self.transform_flag:
            for i in range(3):
                image[:, :, i] = cv2.equalizeHist(image[:, :, i])

        if self.move_flag == 0:
            return image, auxiliary_image
        elif self.move_flag == 1:
            return auxiliary_image, image
        else:
            return image, image.copy()

    def image_update(self):
        image, auxiliary_image = self.transform()

        if len(self.lines) > 0:
            lines = self.lines
            self.camera.insert_line(image, lines, color=[0, 255, 0], thickness=self.line_width)

            pts = self.lines.reshape(-1, 2)
            for pt in pts:
                pt = np.int32(np.round(pt))
                cv2.circle(image, tuple(pt), radius=self.point_radius, color=[0, 0, 255], thickness=-1)

            if self.line_index >= 0 and self.label_endpoint is None:
                lines = self.lines[self.line_index:self.line_index + 1]
                self.camera.insert_line(image, lines, color=[255, 0, 0], thickness=self.line_width)

        if self.label_endpoint is not None:
            pt = np.int32(np.round(self.label_endpoint))
            cv2.circle(image, tuple(pt), radius=self.point_radius, color=[255, 0, 0], thickness=-1)

        if self.capture_endpoint is not None:
            pt = np.int32(np.round(self.capture_endpoint))
            cv2.circle(image, tuple(pt), radius=self.point_select_thresh, color=[255, 0, 255], thickness=-1)

        new_size = (int(round(image.shape[1] * self.scale)), int(round(image.shape[0] * self.scale)))
        image = cv2.resize(image, new_size, cv2.INTER_CUBIC)
        image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3,
                       QImage.Format.Format_BGR888)
        pixmap = QPixmap(image).scaled(new_size[0], new_size[1])
        self.label_Image.setPixmap(pixmap)

        auxiliary_image = cv2.resize(auxiliary_image, new_size, cv2.INTER_CUBIC)
        auxiliary_image = QImage(auxiliary_image.data, auxiliary_image.shape[1], auxiliary_image.shape[0],
                                 auxiliary_image.shape[1] * 3, QImage.Format.Format_BGR888)
        pixmap = QPixmap(auxiliary_image).scaled(new_size[0], new_size[1])
        self.label_AuxiliaryImage.setPixmap(pixmap)

    def widget_update(self):
        # Update Widget
        self.text_File.setText(f'File list: {self.file_index + 1} / {self.num_file}')
        self.list_File.clear()
        self.list_File.addItems(self.file_list)
        self.list_File.setCurrentRow(self.file_index)
        self.list_File.setEnabled(self.save_flag)

        self.button_OpenDir.setEnabled(True)
        self.button_Save.setEnabled(not self.save_flag)
        self.button_Next.setEnabled(self.save_flag and self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.save_flag and self.file_index > 0)
        self.button_Create.setEnabled(True)

        self.menu_OpenDir.setEnabled(True)
        self.menu_Save.setEnabled(not self.save_flag)
        self.menu_Next.setEnabled(self.save_flag and self.file_index < self.num_file - 1)
        self.menu_Prev.setEnabled(self.save_flag and self.file_index > 0)
        self.menu_Create.setEnabled(True)
        self.menu_Transform.setEnabled(True)
        self.menu_Exchange.setEnabled(True)

        self.label_Image.setEnabled(True)

    def line_update(self):
        self.label_endpoint = None
        self.text_Line.setText(f'Line list: {self.line_index + 1} / {len(self.lines)}')
        self.list_Line.clear()
        self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                 self.lines.reshape(-1, 4)])
        self.list_Line.setCurrentRow(self.line_index)
        self.list_Line.setEnabled(len(self.lines) > 0)

        self.button_Delete.setEnabled(len(self.lines) > 0)
        self.menu_Delete.setEnabled(len(self.lines) > 0)

    def zoom_update(self):
        self.text_zoom.setText(f'{int(round(self.scale * 100))}%')
        self.text_zoom.setEnabled(True)
        self.button_ZoomIn.setEnabled(self.scale < self.scale_limit[1])
        self.button_ZoomOut.setEnabled(self.scale > self.scale_limit[0])

    def OpenDir_Callback(self, data_path=None):
        if not data_path:
            data_path = QFileDialog.getExistingDirectory()
        image_path = os.path.join(data_path, self.image_folder)
        if data_path == '' or not os.path.isdir(image_path):
            return

        file_list = []
        for pattern in self.patterns:
            file_list += sorted(glob.glob(os.path.join(image_path, pattern)))
        num_file = len(file_list)
        if num_file == 0:
            return
        label_path = os.path.join(data_path, self.label_folder)
        os.makedirs(label_path, exist_ok=True)

        self.data_path = data_path
        self.file_list = file_list
        self.file_index = 0
        self.num_file = num_file

        self.data_update()
        if self.default_image_size and self.scale == 0:
            scale = min(self.default_image_size[0] / self.image.shape[1],
                        self.default_image_size[1] / self.image.shape[0])
            self.scale = float(np.clip(scale, self.scale_limit[0], self.scale_limit[1]))
            self.text_zoom.setText(f'{int(round(self.scale * 100))}%')

        self.widget_update()
        self.line_update()
        self.zoom_update()
        self.image_update()

    def Save_Callback(self):
        self.save_flag = True
        image_file = self.file_list[self.file_index]
        filename = os.path.basename(image_file)
        filename = os.path.splitext(filename)[0] + '.mat'
        line_file = os.path.join(self.data_path, self.label_folder, filename)
        if len(self.lines):
            sio.savemat(line_file, {'lines': self.lines})
        elif os.path.isfile(line_file):
            os.remove(line_file)

        self.widget_update()

    def Next_Callback(self):
        self.file_index += 1
        self.data_update()
        self.widget_update()
        self.line_update()
        self.image_update()

    def Prev_Callback(self):
        self.file_index -= 1
        self.data_update()
        self.widget_update()
        self.line_update()
        self.image_update()

    def ListFile_Callback(self):
        self.file_index = self.list_File.currentRow()
        self.data_update()
        self.widget_update()
        self.line_update()
        self.image_update()

    def ListLine_Callback(self):
        self.line_index = self.list_Line.currentRow()
        self.line_update()
        self.image_update()

    def Create_Callback(self):
        self.label_flag = True
        self.setCursor(Qt.CrossCursor)

        # Update Widget
        self.reset()
        self.label_Image.setEnabled(True)

    def Delete_Callback(self):
        self.lines = np.delete(self.lines, self.line_index, axis=0)
        self.line_index = len(self.lines) - 1
        self.save_flag = False
        # Update UI
        self.widget_update()
        self.line_update()
        self.image_update()

    def Transform_Callback(self):
        self.transform_flag = not self.transform_flag
        self.image_update()

    def Exchange_Callback(self):
        self.move_flag = (self.move_flag + 1) % 3
        self.image_update()

    def ZoomIn_Callback(self):
        self.scale = float(np.round(self.scale + 0.1, decimals=1))
        self.zoom_update()
        self.image_update()

    def ZoomOut_Callback(self):
        self.scale = float(np.round(self.scale - 0.1, decimals=1))
        self.zoom_update()
        self.image_update()

    def TextZoom_Callback(self):
        text = self.text_zoom.text().strip('%')
        self.scale = np.round(float(text) / 100.0, decimals=1)
        self.scale = float(np.clip(self.scale, self.scale_limit[0], self.scale_limit[1]))
        self.zoom_update()
        self.image_update()

    def mouseRelease_Callback(self, event):
        if event.button() == Qt.RightButton:
            if self.button_Create.isEnabled():
                self.Create_Callback()
            else:
                self.label_flag = False
                self.label_endpoint = None
                self.capture_endpoint = None
                self.setCursor(Qt.ArrowCursor)
                self.widget_update()
                self.line_update()
                self.zoom_update()
                self.image_update()

        elif event.button() == Qt.LeftButton:
            width, height = self.image.shape[1], self.image.shape[0]
            image_width = int(round(width * self.scale))
            image_height = int(round(height * self.scale))
            widget_width = self.label_Image.width()
            widget_height = self.label_Image.height()
            dx, dy = (widget_width - image_width) / 2.0, (widget_height - image_height) / 2.0
            x = np.round((event.x() - dx) / self.scale, decimals=self.decimal_precision)
            y = np.round((event.y() - dy) / self.scale, decimals=self.decimal_precision)
            if x < 0 or x >= width or y < 0 or y >= height:
                return

            pt = np.array([x, y], np.float32)
            if self.label_flag:
                if self.label_endpoint is None:
                    if self.capture_endpoint is not None:
                        self.label_endpoint = self.capture_endpoint
                        self.capture_endpoint = None
                    else:
                        self.label_endpoint = pt

                    # Update UI
                    self.image_update()

                else:
                    if np.abs(self.label_endpoint[0] - pt[0]) <= self.point_align_thresh:
                        pt[0] = self.label_endpoint[0]
                    if np.abs(self.label_endpoint[1] - pt[1]) <= self.point_align_thresh:
                        pt[1] = self.label_endpoint[1]
                    self.label_flag = False
                    if self.capture_endpoint is not None:
                        pt = self.capture_endpoint
                        self.capture_endpoint = None

                    line = np.stack((self.label_endpoint, pt))
                    length = np.linalg.norm(line[1] - line[0], axis=-1)
                    if length > 0:
                        self.lines = np.concatenate((self.lines, line[None]))
                        self.lines[:, :, 0] = np.clip(self.lines[:, :, 0], 0, width - 1e-4)
                        self.lines[:, :, 1] = np.clip(self.lines[:, :, 1], 0, height - 1e-4)
                        self.line_index = len(self.lines) - 1

                    # Update UI
                    self.save_flag = False
                    self.setCursor(Qt.ArrowCursor)
                    self.widget_update()
                    self.line_update()
                    self.zoom_update()
                    self.image_update()
            else:
                if len(self.lines) == 0:
                    return

                dists = []
                pts_list = self.camera.interp_line(self.lines)
                for pts in pts_list:
                    dist = np.linalg.norm(pts - pt[None], axis=-1).min()
                    dists.append(dist)
                dists = np.asarray(dists)
                min_dist = dists.min()
                if min_dist > self.line_select_thresh:
                    return

                self.line_index = dists.argmin()
                self.line_update()
                self.image_update()

    def mouseMove_Callback(self, event):
        last_capture_endpoint = self.capture_endpoint
        self.capture_endpoint = None
        if self.label_flag and len(self.lines) > 0:
            width, height = self.image.shape[1], self.image.shape[0]
            image_width = int(round(width * self.scale))
            image_height = int(round(height * self.scale))
            widget_width = self.label_Image.width()
            widget_height = self.label_Image.height()
            dx, dy = (widget_width - image_width) / 2.0, (widget_height - image_height) / 2.0
            x = np.round((event.x() - dx) / self.scale, decimals=self.decimal_precision)
            y = np.round((event.y() - dy) / self.scale, decimals=self.decimal_precision)
            if 0 <= x < width and 0 <= y < height:
                pt = np.array([x, y], np.float32)
                pts = self.lines.reshape(-1, 2)
                dists = np.linalg.norm(pts - pt[None], axis=-1)
                dist = dists.min()
                if dist <= self.point_select_thresh:
                    index = dists.argmin()
                    self.capture_endpoint = pts[index]

        # Update UI
        if last_capture_endpoint is not None or self.capture_endpoint is not None:
            self.image_update()

    def Tutorial_Callback(self):
        QMessageBox.information(self,
                        'Tutorial',
                        'Version: 1.0\n'
                        'Author: lh9171338\n'
                        'Date: 2022-06-30\n'
                        'Shortcut (default):\n'
                        '\tCtrl + O: Select an image folder\n'
                        '\tCtrl + S: Save the annotations\n'
                        '\tCtrl + V: Go to the next image\n'
                        '\tCtrl + B: Go to the previous image\n'
                        '\tCtrl + C: Create a new annotation\n'
                        '\tCtrl + D: Delete the selected annotation\n'
                        '\tCtrl + W: Transform\n'
                        '\tCtrl + Q: Exchange\n'
                        '\tCtrl + U: View the tutorial\n'
                        'Mouse buttons (clicked in the image area): \n'
                        '\tLeft button: Create an endpoint of a new annotation\n'
                        '\tRight button: Create a new annotation',
                        QMessageBox.Close)


if __name__ == "__main__":
    cfg = CfgNode.load_cfg(open('default.yaml'))
    cfg.freeze()
    print(cfg)

    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    window.OpenDir_Callback()
    sys.exit(app.exec_())

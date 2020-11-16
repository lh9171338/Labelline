import sys
import os
import numpy as np
import glob
import cv2
import PIL.Image as Image
import scipy.io as sio
import argparse
from yacs.config import CfgNode
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import camera as cam


class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()

        # Variables
        self.file_list = []
        self.file_index = 0
        self.num_file = 0
        self.image = None
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.line_index = len(self.lines) - 1
        self.endpoint = None
        self.num_endpoints = 0
        self.capture_endpoint = None
        self.type = cfg.type
        self.coeff_file = cfg.coeff_file
        self.camera = None
        self.decimal_precision = cfg.decimal_precision

        # Parameters
        self.scale = cfg.init_scale
        self.scale_limit = cfg.scale_limit
        self.line_width = cfg.line_width
        self.point_radius = cfg.point_radius
        self.point_select_thresh = 2 * self.point_radius if cfg.point_select_thresh is not None else cfg.point_select_thresh
        self.line_select_thresh = cfg.line_select_thresh
        self.point_vertical_align_thresh = cfg.point_vertical_align_thresh
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
        self.menu_Tutorial = self.menu_Help.addAction(QIcon('icon/tutorial.png'), 'Tutorial')

        self.button_OpenDir = QPushButton('Open Dir', icon=QIcon('icon/open.png'))
        self.button_Save = QPushButton('Save', icon=QIcon('icon/save.png'))
        self.button_Next = QPushButton('Next', icon=QIcon('icon/next.png'))
        self.button_Prev = QPushButton('Prev', icon=QIcon('icon/prev.png'))
        self.button_Create = QPushButton('Create', icon=QIcon('icon/create.png'))
        self.button_Delete = QPushButton('Delete', icon=QIcon('icon/delete.png'))
        self.button_ZoomIn = QPushButton('Zoom In', icon=QIcon('icon/zoom-in.png'))
        self.button_ZoomOut = QPushButton('Zoom Out', icon=QIcon('icon/zoom-out.png'))
        self.text_zoom = QLineEdit('100%', alignment=Qt.AlignCenter)
        self.text_zoom.setValidator(QRegExpValidator(QRegExp('\d+%?')))
        self.text_zoom.setFixedSize(self.button_OpenDir.sizeHint())
        self.buttonLayout = QVBoxLayout()
        self.label_Image = QLabel()
        self.imageLayout = QVBoxLayout()

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
        self.menu_Tutorial.triggered.connect(self.Tutorial_Callback)

        self.menu_OpenDir.setShortcut(cfg.menu_OpenDir_shortcut)
        self.menu_Save.setShortcut(cfg.menu_Save_shortcut)
        self.menu_Next.setShortcut(cfg.menu_Next_shortcut)
        self.menu_Prev.setShortcut(cfg.menu_Prev_shortcut)
        self.menu_Create.setShortcut(cfg.menu_Create_shortcut)
        self.menu_Delete.setShortcut(cfg.menu_Delete_shortcut)
        self.menu_Tutorial.setShortcut(cfg.menu_Tutorial_shortcut)

        self.list_Line.clicked.connect(self.ListLine_Callback)
        self.list_File.clicked.connect(self.ListFile_Callback)
        self.text_zoom.editingFinished.connect(self.TextZoom_Callback)

        self.label_Image.mouseReleaseEvent = self.mouseRelease_Callback
        self.label_Image.mouseMoveEvent = self.mouseMove_Callback
        self.label_Image.setMouseTracking(True)

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(False)
        self.button_Prev.setEnabled(False)
        self.button_Create.setEnabled(False)
        self.button_Delete.setEnabled(False)
        self.button_ZoomIn.setEnabled(False)
        self.button_ZoomOut.setEnabled(False)

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(False)
        self.menu_Prev.setEnabled(False)
        self.menu_Create.setEnabled(False)
        self.menu_Delete.setEnabled(False)

        self.list_Line.setEnabled(False)
        self.list_File.setEnabled(False)
        self.text_zoom.setEnabled(False)

        self.label_Image.setEnabled(False)

    def set_camera(self):
        if cfg.type == 0:
            self.camera = cam.Pinhole()
        elif cfg.type == 1:
            image_file = self.file_list[self.file_index]
            coeff_file = '.'.join(image_file.split('.')[:-1]) + '.yaml'
            if not os.path.isfile(coeff_file):
                coeff_file = self.coeff_file
                if not os.path.isfile(coeff_file):
                    print(f'{coeff_file} does not exist!')
                    exit()
            self.camera = cam.Fisheye()
            self.camera.load_coeff(coeff_file)
        else:
            self.camera = cam.Spherical((self.image.shape[1], self.image.shape[0]))

    def plot(self):
        image = self.image.copy()
        if len(self.lines) > 0:
            try:
                lines = self.camera.truncate_line(self.lines)
            except:
                lines = self.lines
            pts = lines.reshape(-1, 2)
            self.camera.insert_line(image, lines, color=[0, 255, 0], thickness=self.line_width)
            for pt in pts:
                pt = np.int32(np.round(pt))
                cv2.circle(image, tuple(pt), radius=self.point_radius, color=[0, 0, 255], thickness=-1)

            if self.line_index >= 0 and self.endpoint is None:
                self.camera.insert_line(image, lines[self.line_index:self.line_index + 1], color=[255, 0, 0],
                                        thickness=self.line_width)

        if self.endpoint is not None:
            pt = np.int32(np.round(self.endpoint))
            cv2.circle(image, tuple(pt), radius=self.point_radius, color=[255, 0, 0], thickness=-1)

        if self.capture_endpoint is not None:
            pt = np.int32(np.round(self.capture_endpoint))
            cv2.circle(image, tuple(pt), radius=self.point_radius * 2, color=[255, 0, 255], thickness=-1)

        image = Image.fromarray(image[:, :, ::-1])
        new_size = (int(round(image.width * self.scale)), int(round(image.height * self.scale)))
        image = image.resize(new_size, Image.BICUBIC)
        pixmap = image.toqpixmap()
        self.label_Image.resize(image.width, image.height)
        self.label_Image.setPixmap(pixmap)

    def OpenDir_Callback(self):
        dir = QFileDialog.getExistingDirectory()
        if not os.path.isdir(dir):
            return

        file_list = []
        for pattern in self.patterns:
            file_list += sorted(glob.glob(os.path.join(dir, pattern)))
        num_file = len(file_list)
        if num_file == 0:
            return

        self.file_list = file_list
        self.file_index = 0
        self.num_file = num_file
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.endpoint = None
        self.num_endpoints = 0
        self.capture_endpoint = None

        image_file = self.file_list[self.file_index]
        line_file = '.'.join(image_file.split('.')[:-1]) + '.mat'
        self.image = cv2.imread(image_file)
        self.set_camera()
        if os.path.isfile(line_file):
            self.lines = sio.loadmat(line_file)['lines'].reshape(-1, 2, 2)
        self.line_index = len(self.lines) - 1

        # Update UI
        self.plot()
        self.list_Line.clear()
        self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                 self.lines.reshape(-1, 4)])
        self.list_Line.setCurrentRow(self.line_index)
        self.list_File.clear()
        self.list_File.addItems(self.file_list)
        self.list_File.setCurrentRow(self.file_index)

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.file_index > 0)
        self.button_Create.setEnabled(True)
        self.button_Delete.setEnabled(self.line_index >= 0)
        self.button_ZoomIn.setEnabled(self.scale < self.scale_limit[1])
        self.button_ZoomOut.setEnabled(self.scale > self.scale_limit[0])

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(self.file_index < self.num_file)
        self.menu_Prev.setEnabled(self.file_index > 0)
        self.menu_Create.setEnabled(True)
        self.menu_Delete.setEnabled(self.line_index >= 0)

        self.list_Line.setEnabled(True)
        self.list_File.setEnabled(True)
        self.text_zoom.setEnabled(True)

        self.label_Image.setEnabled(True)

    def Save_Callback(self):
        image_file = self.file_list[self.file_index]
        line_file = '.'.join(image_file.split('.')[:-1]) + '.mat'
        try:
            K, D = self.camera.coeff['K'], self.camera.coeff['D']
            sio.savemat(line_file, {'lines': self.lines, 'K': K, 'D': D})
        except:
            sio.savemat(line_file, {'lines': self.lines})

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.file_index > 0)

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(self.file_index < self.num_file)
        self.menu_Prev.setEnabled(self.file_index > 0)

        self.list_File.setEnabled(True)

    def Next_Callback(self):
        self.file_index += 1
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.endpoint = None
        self.num_endpoints = 0

        image_file = self.file_list[self.file_index]
        line_file = '.'.join(image_file.split('.')[:-1]) + '.mat'
        self.image = cv2.imread(image_file)
        self.set_camera()
        if os.path.isfile(line_file):
            self.lines = sio.loadmat(line_file)['lines'].reshape(-1, 2, 2)
        self.line_index = len(self.lines) - 1

        # Update UI
        self.plot()
        self.list_Line.clear()
        self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                 self.lines.reshape(-1, 4)])
        self.list_Line.setCurrentRow(self.line_index)
        self.list_File.setCurrentRow(self.file_index)

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.file_index > 0)
        self.button_Delete.setEnabled(self.line_index >= 0)

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(self.file_index < self.num_file)
        self.menu_Prev.setEnabled(self.file_index > 0)
        self.menu_Delete.setEnabled(self.line_index >= 0)

    def Prev_Callback(self):
        self.file_index -= 1
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.endpoint = None
        self.num_endpoints = 0

        image_file = self.file_list[self.file_index]
        line_file = '.'.join(image_file.split('.')[:-1]) + '.mat'
        self.image = cv2.imread(image_file)
        self.set_camera()
        if os.path.isfile(line_file):
            self.lines = sio.loadmat(line_file)['lines'].reshape(-1, 2, 2)
        self.line_index = len(self.lines) - 1

        # Update UI
        self.plot()
        self.list_Line.clear()
        self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                 self.lines.reshape(-1, 4)])
        self.list_Line.setCurrentRow(self.line_index)
        self.list_File.setCurrentRow(self.file_index)

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.file_index > 0)
        self.button_Delete.setEnabled(self.line_index >= 0)

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(self.file_index < self.num_file)
        self.menu_Prev.setEnabled(self.file_index > 0)
        self.menu_Delete.setEnabled(self.line_index >= 0)

    def Create_Callback(self):
        self.num_endpoints = 1

        # Update UI
        self.setCursor(Qt.CrossCursor)

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(False)
        self.button_Prev.setEnabled(False)
        self.button_Create.setEnabled(False)
        self.button_Delete.setEnabled(False)

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(False)
        self.menu_Prev.setEnabled(False)
        self.menu_Create.setEnabled(False)
        self.menu_Delete.setEnabled(False)

        self.list_Line.setEnabled(False)
        self.list_File.setEnabled(False)

    def Delete_Callback(self):
        self.lines = np.delete(self.lines, self.line_index, axis=0)
        self.line_index = len(self.lines) - 1

        # Update UI
        self.plot()
        self.list_Line.clear()
        self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                 self.lines.reshape(-1, 4)])
        self.list_Line.setCurrentRow(self.line_index)

        # Update Widget
        self.button_Save.setEnabled(True)
        self.button_Next.setEnabled(False)
        self.button_Prev.setEnabled(False)
        self.button_Delete.setEnabled(self.line_index >= 0)

        self.menu_Save.setEnabled(True)
        self.menu_Next.setEnabled(False)
        self.menu_Prev.setEnabled(False)
        self.menu_Delete.setEnabled(self.line_index >= 0)

        self.list_File.setEnabled(False)

    def ZoomIn_Callback(self):
        self.scale = np.round(self.scale + 0.1, decimals=1)

        # Update UI
        self.plot()
        self.text_zoom.setText(f'{int(round(self.scale * 100))}%')

        # Update Widget
        self.button_ZoomIn.setEnabled(self.scale < self.scale_limit[1])
        self.button_ZoomOut.setEnabled(self.scale > self.scale_limit[0])

    def ZoomOut_Callback(self):
        self.scale = np.round(self.scale - 0.1, decimals=1)

        # Update UI
        self.plot()
        self.text_zoom.setText(f'{int(round(self.scale * 100))}%')

        # Update Widget
        self.button_ZoomIn.setEnabled(self.scale < self.scale_limit[1])
        self.button_ZoomOut.setEnabled(self.scale > self.scale_limit[0])

    def Tutorial_Callback(self):
        QMessageBox.information(self,
                        'Tutorial',
                        'Version: 1.0\n'
                        'Author: lh9171338\n'
                        'Date: 2020-11-07\n'
                        'Shortcut (default):\n'
                        '\tCtrl + O: Select an image folder\n'
                        '\tCtrl + S: Save the annotations\n'
                        '\tCtrl + V: Go to the next image\n'
                        '\tCtrl + B: Go to the previous image\n'
                        '\tCtrl + C: Create a new annotation\n'
                        '\tCtrl + D: Delete a selected annotation\n'
                        '\tCtrl + U: View the tutorial\n'
                        'Mouse buttons (clicked in the image area): \n'
                        '\tLeft button: Create a new endpoint\n'
                        '\tRight button: Create a new annotation',
                        QMessageBox.Close)

    def ListLine_Callback(self):
        self.line_index = self.list_Line.currentRow()
        self.capture_endpoint = None

        # Update UI
        self.plot()

    def ListFile_Callback(self):
        self.file_index = self.list_File.currentRow()
        self.lines = np.zeros((0, 2, 2), np.float32)
        self.endpoint = None
        self.num_endpoints = 0

        image_file = self.file_list[self.file_index]
        line_file = '.'.join(image_file.split('.')[:-1]) + '.mat'
        self.image = cv2.imread(image_file)
        self.set_camera()
        if os.path.isfile(line_file):
            self.lines = sio.loadmat(line_file)['lines'].reshape(-1, 2, 2)
        self.line_index = len(self.lines) - 1

        # Update UI
        self.plot()
        self.list_Line.clear()
        self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                 self.lines.reshape(-1, 4)])
        self.list_Line.setCurrentRow(self.line_index)
        self.list_File.setCurrentRow(self.file_index)

        # Update Widget
        self.button_Save.setEnabled(False)
        self.button_Next.setEnabled(self.file_index < self.num_file - 1)
        self.button_Prev.setEnabled(self.file_index > 0)
        self.button_Delete.setEnabled(self.line_index >= 0)

        self.menu_Save.setEnabled(False)
        self.menu_Next.setEnabled(self.file_index < self.num_file - 1)
        self.menu_Prev.setEnabled(self.file_index > 0)
        self.menu_Delete.setEnabled(self.line_index >= 0)

    def TextZoom_Callback(self):
        text = self.text_zoom.text().strip('%')
        self.scale = np.round(float(text) / 100.0, decimals=1)
        self.scale = np.clip(self.scale, self.scale_limit[0], self.scale_limit[1])

        # Update UI
        self.plot()
        self.text_zoom.setText(f'{int(round(self.scale * 100))}%')

        # Update Widget
        self.button_ZoomIn.setEnabled(self.scale < self.scale_limit[1])
        self.button_ZoomOut.setEnabled(self.scale > self.scale_limit[0])

    def mouseRelease_Callback(self, event):
        self.capture_endpoint = None
        if event.button() == Qt.RightButton:
            if self.button_Create.isEnabled():
                self.Create_Callback()

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
            if self.num_endpoints == 0:
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
                self.list_Line.setCurrentRow(self.line_index)

                # Update UI
                self.plot()

            elif self.num_endpoints == 1:
                self.num_endpoints = 2
                if len(self.lines) > 0:
                    pts = self.lines.reshape(-1, 2)
                    dists = np.linalg.norm(pts - pt[None], axis=-1)
                    dist = dists.min()
                    if dist <= self.point_select_thresh:
                        index = dists.argmin()
                        pt = pts[index]
                self.endpoint = pt

                # Update UI
                self.plot()

            else:
                self.num_endpoints = 0
                if abs(pt[0] - self.endpoint[0]) <= self.point_vertical_align_thresh:
                    pt[0] = self.endpoint[0]
                if len(self.lines) > 0:
                    pts = self.lines.reshape(-1, 2)
                    dists = np.linalg.norm(pts - pt[None], axis=-1)
                    dist = dists.min()
                    if dist <= self.point_select_thresh:
                        index = dists.argmin()
                        pt = pts[index]

                line = np.concatenate((self.endpoint[None], pt[None]))
                self.lines = np.concatenate((self.lines, line[None]))
                self.line_index = len(self.lines) - 1
                self.endpoint = None

                # Update UI
                self.plot()
                self.setCursor(Qt.ArrowCursor)
                self.list_Line.clear()
                self.list_Line.addItems([f'[{line[0]:.3f}, {line[1]:.3f}, {line[2]:.3f}, {line[3]:.3f}]' for line in
                                         self.lines.reshape(-1, 4)])
                self.list_Line.setCurrentRow(self.line_index)

                # Update Widget
                self.button_Save.setEnabled(True)
                self.button_Create.setEnabled(True)
                self.button_Delete.setEnabled(self.line_index >= 0)

                self.menu_Save.setEnabled(True)
                self.menu_Create.setEnabled(True)
                self.menu_Delete.setEnabled(self.line_index >= 0)

                self.list_Line.setEnabled(True)

    def mouseMove_Callback(self, event):
        self.capture_endpoint = None
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
            if self.num_endpoints > 0:
                if len(self.lines) > 0:
                    pts = self.lines.reshape(-1, 2)
                    dists = np.linalg.norm(pts - pt[None], axis=-1)
                    dist = dists.min()
                    if dist <= self.point_select_thresh:
                        index = dists.argmin()
                        self.capture_endpoint = pts[index]

        # Update UI
        self.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=int, choices=[0, 1, 2],
                        help='0: pinhole image, 1: fisheye image, 2: spherical image', required=True)
    parser.add_argument('-c', '--coeff_file', type=str, help='fisheye camera distortion coefficients file')
    opts = parser.parse_args()
    opts_dict = vars(opts)
    opts_list = []
    for key, value in zip(opts_dict.keys(), opts_dict.values()):
        if value is not None:
            opts_list.append(key)
            opts_list.append(value)
    cfg = CfgNode.load_cfg(open('default.yaml'))
    cfg.merge_from_list(opts_list)
    cfg.freeze()
    print(cfg)

    app = QApplication(sys.argv)
    window = MainWindow(cfg)
    window.show()
    sys.exit(app.exec_())

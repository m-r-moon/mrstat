#!/usr/local/bin/python3

import csv, math, random, sys

from PyQt5.QtCore import QAbstractTableModel, QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QPalette, QPixmap
from PyQt5.QtWidgets import QAction, QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPushButton, QSizePolicy, QSpinBox, QStackedLayout, QStatusBar, QStyle, QTabWidget, QTableView, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget

from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg


class App(QMainWindow):

  def __init__(self):
    super().__init__()
    self.left = 10
    self.top = 10
    self.title = 'MR Stats'
    self.width = 1280
    self.height = 720
    self.initUI()

  def initUI(self):
    self.setWindowTitle(self.title)
    self.setGeometry(self.left, self.top, self.width, self.height)

    self.labels, self.x_vals, self.y_vals, self.file_data, self.anova_data, self.anova_headers, self.anova_rows, self.other_data, self.other_headers, self.other_rows = [], [], [], [], [], [], [], [], [], []
    self.x_col, self.y_col = None, None
    self.labels_present = False

    app_layout = QHBoxLayout()

    form_layout = QVBoxLayout()
    form_layout.addStretch(1)

    self.btn_open = QPushButton('Open')
    self.btn_open.pressed.connect(self.btn_open_click)
    form_layout.addWidget(self.btn_open)
    form_layout.addStretch(1)

    form_layout.addWidget(QLabel('Method'))
    self.cmb_method = QComboBox()
    self.cmb_method.addItems(['Least Squares', 'Moving Average', 'Exponential Smoothing', 'Trend Projection'])
    self.cmb_method.currentIndexChanged.connect(self.cmb_method_index_changed)
    self.cmb_method.setEnabled(False)
    form_layout.addWidget(self.cmb_method)

    param_layout = QVBoxLayout()

    param_layout.addWidget(QLabel('Alpha'))
    self.dspn_alpha = QDoubleSpinBox()
    self.dspn_alpha.setMinimum(0.01)
    self.dspn_alpha.setMaximum(0.99)
    self.dspn_alpha.setSingleStep(0.01)
    self.dspn_alpha.valueChanged.connect(self.dspn_alpha_changed)
    self.dspn_alpha.setEnabled(False)
    param_layout.addWidget(self.dspn_alpha)

    param_layout.addWidget(QLabel('Gamma'))
    self.dspn_gamma = QDoubleSpinBox()
    self.dspn_gamma.setMinimum(0.01)
    self.dspn_gamma.setMaximum(10.00)
    self.dspn_gamma.setSingleStep(0.01)
    self.dspn_gamma.valueChanged.connect(self.dspn_gamma_changed)
    self.dspn_gamma.setEnabled(False)
    param_layout.addWidget(self.dspn_gamma)

    param_layout.addWidget(QLabel('Beta'))
    self.dspn_beta = QDoubleSpinBox()
    self.dspn_beta.setMinimum(0.01)
    self.dspn_beta.setMaximum(30.00)
    self.dspn_beta.setSingleStep(0.01)
    self.dspn_beta.valueChanged.connect(self.dspn_beta_changed)
    self.dspn_beta.setEnabled(False)
    param_layout.addWidget(self.dspn_beta)

    param_layout.addWidget(QLabel('Periods'))
    self.spn_periods = QSpinBox()
    self.spn_periods.setMinimum(2)
    self.spn_periods.setMaximum(3)
    self.spn_periods.setSingleStep(1)
    self.spn_periods.valueChanged.connect(self.spn_periods_changed)
    self.spn_periods.setEnabled(False)
    param_layout.addWidget(self.spn_periods)

    form_layout.addWidget(QLabel('Parameters'))
    frame_layout = QFrame()
    frame_layout.setLayout(param_layout)
    form_layout.addWidget(frame_layout)
    form_layout.addStretch(2)

    self.btn_scatterplot = QPushButton('Scatter Plot')
    self.btn_scatterplot.pressed.connect(self.btn_scatterplot_click)
    self.btn_scatterplot.setEnabled(False)
    form_layout.addWidget(self.btn_scatterplot)

    self.btn_trendplot = QPushButton('Trend Plot')
    self.btn_trendplot.pressed.connect(self.btn_trendplot_click)
    self.btn_trendplot.setEnabled(False)
    form_layout.addWidget(self.btn_trendplot)

    self.btn_analyze = QPushButton('Analyze')
    self.btn_analyze.pressed.connect(self.btn_analyze_click)
    self.btn_analyze.setEnabled(False)
    form_layout.addWidget(self.btn_analyze)

    self.btn_clear = QPushButton('Reset')
    self.btn_clear.pressed.connect(self.btn_clear_click)
    form_layout.addWidget(self.btn_clear)

    app_layout.addLayout(form_layout, 2)

    data_layout = QVBoxLayout()

    tabs_layout = QTabWidget()

    data_tab = QWidget()
    tabs_layout.addTab(data_tab, 'Data')

    data_tab_layout = QVBoxLayout()

    self.table = QTableView()
    self.model = Table([[]])
    self.table.setModel(self.model)
    data_tab_layout.addWidget(self.table)
    data_tab.setLayout(data_tab_layout)
    
    graph_tab = QWidget()
    tabs_layout.addTab(graph_tab, 'Graph')

    graph_tab_layout = QVBoxLayout()

    self.graph = pg.PlotWidget()
    graph_tab_layout.addWidget(self.graph)
    graph_tab.setLayout(graph_tab_layout)

    anova_tab = QWidget()
    tabs_layout.addTab(anova_tab, 'ANOVA')

    self.anova_tab_layout = QVBoxLayout()

    self.anova = QTableView()
    self.anova_model = Table([[]])
    self.anova.setModel(self.anova_model)

    self.anova_tab_layout.addWidget(self.anova)
    anova_tab.setLayout(self.anova_tab_layout)

    analysis_tab = QWidget()
    tabs_layout.addTab(analysis_tab, 'Analysis')

    self.analysis_tab_layout = QVBoxLayout()

    self.analysis = QTableView()
    self.analysis_model = Table([[]])
    self.analysis.setModel(self.analysis_model)

    self.analysis_tab_layout.addWidget(self.analysis)
    analysis_tab.setLayout(self.analysis_tab_layout)

    data_layout.addWidget(tabs_layout)

    app_layout.addLayout(data_layout, 6)

    widget = QWidget()
    widget.setLayout(app_layout)
    self.setCentralWidget(widget)

    self.show()

  def dialog_yn(self, title, text, icon):
    dlg = QMessageBox(self)
    dlg.setWindowTitle(title)
    dlg.setText(text)
    dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    dlg.setIcon(icon)
    resp = dlg.exec()

    if resp == QMessageBox.Yes:
      return True
    else:
      return False

  def dialog_alert(self, title, text, icon):
    dlg = QMessageBox(self)
    dlg.setWindowTitle(title)
    dlg.setText(text)
    dlg.setStandardButtons(QMessageBox.Ok)
    dlg.setIcon(icon)
    dlg.exec()

  def fill_data_table(self):
    self.model = Table(self.file_data, self.labels)
    self.table.setModel(self.model)

  def btn_open_click(self):
    print('open button clicked')
    file_name, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "CSV Files (*.csv)")
    if file_name:
      print(file_name)
      self.labels_present = self.dialog_yn('Labels present?', 'Are labels present in the dataset?', QMessageBox.Question)
      print(self.labels_present)
      with open(file_name, 'r') as dfp:
        csv_data = list(csv.reader(dfp))
        if self.labels_present:
          print('we have labels')
          self.labels = csv_data[0]
          csv_data.pop(0)
        else:
          print('no labels')
          self.labels = [i + 1 for i in range(len(csv_data[0]))]
        self.file_data = csv_data
        self.fill_data_table()
        self.cmb_method.setEnabled(True)
        self.btn_scatterplot.setEnabled(True)
        self.btn_trendplot.setEnabled(True)
        self.spn_periods.setMaximum(len(self.file_data) - 1)

  def cmb_method_index_changed(self, idx):
    method = str(self.cmb_method.currentText())
    print('method: ' + method)
    if method == 'Least Squares':
      self.dspn_alpha.setEnabled(False)
      self.dspn_gamma.setEnabled(False)
      self.dspn_beta.setEnabled(False)
      self.spn_periods.setEnabled(False)
    elif method == 'Moving Average':
      self.dspn_alpha.setEnabled(False)
      self.dspn_gamma.setEnabled(False)
      self.dspn_beta.setEnabled(False)
      self.spn_periods.setEnabled(True)
    elif method == 'Exponential Smoothing':
      self.dspn_alpha.setEnabled(True)
      self.dspn_gamma.setEnabled(False)
      self.dspn_beta.setEnabled(False)
      self.spn_periods.setEnabled(False)
    elif method == 'Trend Projection':
      self.dspn_alpha.setEnabled(False)
      self.dspn_gamma.setEnabled(False)
      self.dspn_beta.setEnabled(False)
      self.spn_periods.setEnabled(False)
    else:
      print('not sure how to proceed...')

  def dspn_alpha_changed(self, idx):
    print('test')
    print('index: ' + str(idx))

  def dspn_gamma_changed(self, idx):
    print('test')
    print('index: ' + str(idx))

  def dspn_beta_changed(self, idx):
    print('test')
    print('index: ' + str(idx))
  
  def spn_periods_changed(self, idx):
    print('test')
    print('index: ' + str(idx))

  def btn_scatterplot_click(self):
    print('test')
    if self.file_data:
      self.x_vals = [int(self.file_data[i][len(self.labels) - 2]) for i in range(len(self.file_data))]
      self.y_vals = [int(self.file_data[i][len(self.labels) - 1]) for i in range(len(self.file_data))]
      self.graph.setLabel('left', self.labels[len(self.labels) - 2])
      self.graph.setLabel('bottom', self.labels[len(self.labels) - 1])
      self.graph.addLegend()
      self.graph.plot(self.x_vals, self.y_vals, pen=None, symbol='+', symbolBrush=('r'))
      self.btn_analyze.setEnabled(True)
    else:
      self.dialog_alert('Error!', 'No dataset available to process.', QMessageBox.Question)

  def btn_trendplot_click(self):
    print('test')
    if self.file_data:
      self.x_vals = [int(self.file_data[i][len(self.labels) - 2]) for i in range(len(self.file_data))]
      self.y_vals = [int(self.file_data[i][len(self.labels) - 1]) for i in range(len(self.file_data))]
      self.graph.setLabel('left', self.labels[len(self.labels) - 2])
      self.graph.setLabel('bottom', self.labels[len(self.labels) - 1])
      self.graph.addLegend()
      self.graph.plot(self.x_vals, self.y_vals, pen=pg.mkPen(color='r'))
      self.btn_analyze.setEnabled(True)
    else:
      self.dialog_alert('Error!', 'No dataset available to process.', QMessageBox.Question)

  def btn_analyze_click(self):
    print('test')
    method = str(self.cmb_method.currentText())
    if method == 'Least Squares':
      self.calc_least_squares()
    elif method == 'Moving Average':
      self.calc_mv_avg()
    elif method == 'Exponential Smoothing':
      self.calc_exp_smoothing()
    elif method == 'Trend Projection':
      self.calc_trend_projection()
    else:
      print('not yet...')
  
  def btn_clear_click(self):
    print('test')

  def calc_least_squares(self):
    sum_x, sum_y = 0, 0
    for idx in range(len(self.x_vals)):
      sum_x += self.x_vals[idx]
      sum_y += self.y_vals[idx]
    x_bar = sum_x / len(self.x_vals)
    y_bar = sum_y / len(self.y_vals)
    x_errs, y_errs, b1_num, b1_den = [], [], [], []
    n_sum, d_sum = 0, 0
    for idx in range(len(self.x_vals)):
      x_errs.append(self.x_vals[idx] - x_bar)
      y_errs.append(self.y_vals[idx] - y_bar)
      b1_num.append(x_errs[idx] * y_errs[idx])
      b1_den.append(pow(x_errs[idx], 2))
      n_sum += b1_num[idx]
      d_sum += b1_den[idx]
    b1 = n_sum / d_sum
    b0 = y_bar - (b1 * x_bar)
    print('line formula: y = ' + str(b0) + ' + ' + str(b1) + 'x')
    pred_sales, err, sqr_err, dev, sqr_dev = [], [], [], [], []
    sse, sst = 0, 0
    for idx in range(len(self.x_vals)):
      pred_sales.append(int(b0 + b1 * self.x_vals[idx]))
      err.append(self.y_vals[idx] - pred_sales[idx])
      sqr_err.append(pow(err[idx], 2))
      sse += sqr_err[idx]
      dev.append(self.y_vals[idx] - y_bar)
      sqr_dev.append(pow(dev[idx], 2))
      sst += sqr_dev[idx]
    print('sse: ' + str(sse) + ' sst: ' + str(sst))
    ssr = sst - sse
    print('ssr = sst - sse')
    print('ssr: ' + str(ssr))
    cod = ssr / sst
    print('cod = ssr / sst')
    print('cod: ' + str(cod))
    cc = math.sqrt(cod)
    cc_sign = '+' if b1 > 0 else '-'
    print('cc: ' + cc_sign + str(cc))
    mse = sse / (len(self.x_vals) - 2)
    print('mse: ' + str(mse))
    see = math.sqrt(mse)
    print('s: ' + str(see))
    x_max = max(self.x_vals)
    x_max = (round(x_max, -1) if round(x_max, -1) > x_max else round(x_max, -1) + 10) + 10
    reg_line_x, reg_line_y, y_bar_line = [], [], []
    for idx in range(0, x_max, 10):
      reg_line_x.append(idx)
      reg_line_y.append(int(b0 + b1 * idx))
      y_bar_line.append(y_bar)
    self.graph.plot(reg_line_x, reg_line_y, pen=pg.mkPen(color='r'))
    self.graph.plot(reg_line_x, y_bar_line, pen=pg.mkPen(color='b'))
    self.graph.setX(0)
    self.anova_data = [[ssr, 1, (ssr / 1), ((ssr / 1) / mse), ''], [sse, (len(self.x_vals) - 2), mse, '', ''], [sst, (len(self.x_vals) - 1), '', '', '']]
    self.analyze_anova()

  def calc_mv_avg(self):
    print('we will handle moving averages here...')
    forecast_data = []
    err = 0.0
    period = int(self.spn_periods.value())
    for idx in range(period):
      forecast_data.append([idx + 1, self.y_vals[0], 0, 0, 0])
    for idx in range(period, len(self.y_vals)):
      f = round((self.y_vals[idx - period] + self.y_vals[idx - period + 1] + self.y_vals[idx - period + 2]) / period)
      e = self.y_vals[idx] - f
      forecast_data.append([idx + 1, self.y_vals[idx], f, e, pow(e, 2)])
      err += pow(e, 2)
      print('we will do something...')
    print(err)
    #print(forecast_data)
    self.other_headers = ['Week', 'Time Series Value', 'Forecast', 'Forecast Error', 'Squared Forecast Error']
    self.other_rows = []
    self.other_data = forecast_data
    self.analyze_other()

  def calc_exp_smoothing(self):
    print('we will handle exponential smoothing here...')
    forecast_data = [[1, self.y_vals[0], 0, 0, 0], [2, self.y_vals[1], self.y_vals[0], self.y_vals[1] - self.y_vals[0], pow(self.y_vals[1] - self.y_vals[0], 2)]]
    err = pow(self.y_vals[1] - self.y_vals[0], 2)
    alpha = float(self.dspn_alpha.value())
    for idx in range(2, len(self.y_vals)):
      f = round((alpha * self.y_vals[idx - 1]) + ((1 - alpha) * forecast_data[idx - 1][2]), 2)
      e = round(self.y_vals[idx] - f, 2)
      forecast_data.append([idx + 1, self.y_vals[idx], f, e, round(pow(e, 2), 2)])
      err += round(pow(e, 2), 2)
    print(err)
    #print(forecast_data)
    self.other_headers = ['Week', 'Time Series Value', 'Forecast', 'Forecast Error', 'Squared Forecast Error']
    self.other_rows = []
    self.other_data = forecast_data
    self.analyze_other()

  def calc_trend_projection(self):
    self.other_data = []
    sum_t, t_bar, sum_Y, Y_bar, sum_t2, sumtY, b1, b0 = 0, 0, 0, 0, 0, 0, 0, 0
    for idx in range(len(self.y_vals)):
      t = idx + 1
      sum_t += t
      Yt = self.y_vals[idx]
      sum_Y += Yt
      tYt = t * Yt
      sum_tY += tYt
      t2 = pow(t, 2)
      sum_t2 += t2
      self.other_data.append([t, Yt, tYt, t2])
    t_bar = sum_t / len(self.y_vals)
    Y_bar = sum_Y / len(self.y_vals)
    b1 = (sum_tY - (sum_t * sum_Y) / len(self.y_vals)) / (sum_t2 - pow(sum_t, 2) / len(self.y_vals))
    b0 = (Y_bar - (b1 * t_bar))
    self.other_data.append([sum_t, sum_Y, sum_tY, sum_t2])
    self.other_data.append([t_bar, Y_bar, b1, b0])
    self.other_data.append([(len(self.y_vals) + 1), (b0 + (b1 * (len(self.y_vals) + 1))), 0, 0])
    self.other_headers = ['t', 'Yt', 'tYt', 't2']
    self.analyze_other()

  def analyze_anova(self):
    print('anova analysis data...')
    self.anova_headers = ['Sum of Squares', 'Degrees of Freedom', 'Mean Square', 'F', 'p-value']
    self.anova_rows = ['Regression', 'Error', 'Total']
    self.anova_model = Table(self.anova_data, self.anova_headers, self.anova_rows)
    self.anova.setModel(self.anova_model)

  def analyze_other(self):
    print('other analysis data...')
    self.analysis_model = Table(self.other_data, self.other_headers, self.other_rows)
    self.analysis.setModel(self.analysis_model)


class Table(QAbstractTableModel):

  def __init__(self, data, headers = [], row_labels = {}):
    super(Table, self).__init__()
    self._data = data
    self._headers = headers
    self._row_labels = row_labels

  def data(self, index, role):
    if role == Qt.DisplayRole:
      return self._data[index.row()][index.column()]

  def rowCount(self, index):
    return len(self._data)

  def columnCount(self, index):
    return len(self._data[0])

  def headerData(self, section, orientation, role):
    if role == Qt.DisplayRole and orientation == Qt.Horizontal:
      return str(self._headers[section])
    elif role == Qt.DisplayRole and orientation == Qt.Vertical:
      return section + 1 if len(self._row_labels) == 0 else self._row_labels[section]


if __name__ == '__main__':
  app = QApplication(sys.argv)
  ex = App()
  sys.exit(app.exec_())


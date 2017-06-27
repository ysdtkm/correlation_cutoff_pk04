#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, textwrap, datetime, sys, subprocess
sys.path.append("../Py")
import const

def main():
  git_output = str(subprocess.check_output("git show HEAD | head -n1", shell=True))
  last_commit = git_output.split()[1][:7]
  date = datetime.datetime.now()
  datestr = "%04d%02d%02d_%02d%02d" % (date.year, date.month, date.day, date.hour, date.minute)
  date_for_tex = datestr.replace("_", "\\_")
  txt_out = header(date_for_tex, last_commit)

  # rmse and lyapunov exponents
  txt_out += figure_single(date_for_tex, last_commit)
  txt_out += write_txt(date_for_tex, last_commit, "../image/true/rmse.txt", "9pt")
  txt_out += write_txt(date_for_tex, last_commit, "../data/lyapunov.txt", "5.4pt")

  # rmse-spread
  tmax = const.STEPS
  explist = ["etkf", "tdvar", "fdvar"]
  filebase = "../image/@@expname@@_@@yname@@_int8/@@expname@@_@@yname@@_int8_@@xname@@_time.png"
  for expname in explist:
    xlist = ["extro", "trop", "ocn"]
    ylist = ["strong", "weak", "non"]
    txt_out += figure_table(expname, filebase, xlist, ylist, date_for_tex, last_commit)

  # time series
  # filebase = "../image/@@expname@@_@@yname@@_int8/@@expname@@_@@yname@@_int8_@@xname@@_val.png"
  # for expname in explist:
  #   xlist = ["extro", "trop", "ocn"]
  #   ylist = ["strong", "weak", "non"]
  #   txt_out += figure_table(expname, filebase, xlist, ylist, date_for_tex, last_commit)

  # all figures
  filebase = "../image/@@expname@@/@@expname@@_@@imgname@@.png"
  for exp in const.EXPLIST:
    txt_out += figure_all(exp["name"], filebase, date_for_tex, last_commit)

  txt_out += footer()
  f = io.open("./temp.tex", "w")
  f.write(txt_out)
  f.close()
  os.system("make")
  os.system("mkdir -p ./archive")
  os.system("cp -f ./out.pdf ./archive/%s_%s_%dsteps.pdf" % (datestr, last_commit, tmax))
  return 0

def header(date, last_commit):
  header = """
    \\documentclass{beamer}
    \\usepackage{xltxtra}
    \\usepackage{amsmath}
    \\usepackage{verbatim}
    \\usepackage[subrefformat=parens]{subcaption}
    \\usetheme{Boadilla}
    \\usecolortheme{beaver}
    \\title{@@title@@}
    \\date{@@date@@}
    \\author{Takuma Yoshida}
    \\parskip=12pt
    \\parindent=0pt

    %gets rid of bottom navigation bars, navigation symbols and footer
    \\setbeamertemplate{footline}[frame number]{}
    \\setbeamertemplate{navigation symbols}{}
    \\setbeamertemplate{footline}{}

    \\newcommand{\\includegraphicsmaybe}[2][width=10mm]{
      \\IfFileExists{#2}{\\includegraphics[#1]{#2}}{\\includegraphics[#1]{dummy.png}
    }}


    \\begin{document}
    % \\maketitle
  """
  header = header.replace('@@date@@', date)
  header = header.replace('@@title@@', last_commit)
  return textwrap.dedent(header[1:-1])

def figure_table(expname, filebase, xlist, ylist, date, commit):
  content = """
    \\begin{frame}
    \\frametitle{@@expname@@}
    \\vspace*{-10mm}
    \\begin{figure}[h]
      \\flushleft
  """[1:-1]
  content = content.replace('@@expname@@', date + " " + commit + " "  + expname)

  for yname in ylist:
    for xname in xlist:
      tex_figure = """
        \\begin{minipage}[b]{0.32\\linewidth}
          \\centering
          \\includegraphicsmaybe[width=36mm]{@@filebase@@}
          % \\subcaption{@@num@@}
        \\end{minipage}
        """[1:-1]
      tex_figure = tex_figure.replace('@@filebase@@', filebase)
      tex_figure = tex_figure.replace('@@expname@@', expname)
      tex_figure = tex_figure.replace('@@xname@@', xname)
      tex_figure = tex_figure.replace('@@yname@@', yname)
      content += tex_figure
    content += "\\\\"

  closing = """
    % \\caption{polygons}\\label{reg_poly}
    \\end{figure}
    \\end{frame}
  """[1:-1]
  content += closing

  return textwrap.dedent(content[1:-1])

def figure_single(date, commit):
  content = """
    \\begin{frame}
    \\frametitle{@@title@@}
    \\vspace*{-10mm}
    \\begin{figure}[h]
      \\flushleft
  """[1:-1]
  content = content.replace('@@title@@', date + " " + commit)

  tex_figure = """
    \\centering
    \\includegraphicsmaybe[width=78mm]{@@filebase@@}
    """[1:-1]
  tex_figure = tex_figure.replace('@@filebase@@', "../image/true/rmse_bar.png")
  content += tex_figure

  closing = """
    \\end{figure}
    \\end{frame}
  """[1:-1]
  content += closing

  return textwrap.dedent(content[1:-1])

def figure_all(expname, filebase, date, commit):
  content = """
    \\begin{frame}
    \\frametitle{@@expname@@}
    \\vspace*{-10mm}
    \\begin{figure}[h]
      \\flushleft
  """[1:-1]
  content = content.replace('@@expname@@', date + " " + commit + " "  + expname.replace("_", "\\_"))

  imglist = ["anl_covar_logrms", "back_covar_logrms", "anl_covar_mean", "back_covar_mean", \
             "extro_traj", "trop_traj", "ocn_traj", "", \
             "extro_val", "trop_val", "ocn_val", "", \
             "extro_time", "trop_time", "ocn_time", ""]
  for i, imgname in enumerate(imglist):
    tex_figure = """
      \\begin{minipage}[b]{0.24\\linewidth}
        \\centering
        \\includegraphicsmaybe[width=27mm]{@@filebase@@}
      \\end{minipage}
      """[1:-1]
    tex_figure = tex_figure.replace('@@filebase@@', filebase)
    tex_figure = tex_figure.replace('@@expname@@', expname)
    tex_figure = tex_figure.replace('@@imgname@@', imgname)
    content += tex_figure
    if i % 4 == 3:
      content += "\\\\"

  closing = """
    \\end{figure}
    \\end{frame}
  """[1:-1]
  content += closing

  return textwrap.dedent(content[1:-1])

def write_txt(date, commit, path, size):
  if not os.path.isfile(path):
    return ""

  content = """
    \\begin{frame}
    \\frametitle{@@expname@@}
    \\fontsize{@@size@@}{@@size@@}{
      \\verbatiminput{@@filepath@@}
    }
    \\end{frame}
  """[1:-1]
  content = content.replace('@@expname@@', date + " " + commit + " rmse")
  content = content.replace('@@filepath@@', path)
  content = content.replace('@@size@@', size)

  return textwrap.dedent(content[1:-1])

def footer():
  footer = """
    \\end{document}
  """
  return textwrap.dedent(footer[1:-1])

main()

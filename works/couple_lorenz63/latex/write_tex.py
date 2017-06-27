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
  txt_out += figure_table("", ["../image/true/rmse_bar.png"], 1, 1, date_for_tex, last_commit)
  txt_out += write_txt(date_for_tex, last_commit, "../image/true/rmse.txt", "9pt")
  txt_out += write_txt(date_for_tex, last_commit, "../data/lyapunov.txt", "5.4pt")

  # rmse-spread comparison
  for expname in ["etkf", "tdvar", "fdvar"]:
    filelist = []
    for strx in ["extro", "trop", "ocn"]:
      for stry in ["strong", "weak", "non"]:
        filelist.append("../image/%s_%s_int8/%s_%s_int8_%s_time.png" % (expname, stry, expname, stry, strx))
    txt_out += figure_table(expname, filelist, 3, 3, date_for_tex, last_commit)

  # all figures
  for exp in const.EXPLIST:
    filelist = []
    imglist = ["anl_covar_logrms", "back_covar_logrms", "anl_covar_mean", "back_covar_mean", \
               "extro_traj", "trop_traj", "ocn_traj", "", \
               "extro_val", "trop_val", "ocn_val", "", \
               "extro_time", "trop_time", "ocn_time", ""]
    for imgname in imglist:
      filelist.append("../image/%s/%s_%s.png" % (exp["name"], exp["name"], imgname))
    txt_out += figure_table(exp["name"], filelist, 4, 4, date_for_tex, last_commit)

  # conditions
  txt_out += write_txt(date_for_tex, last_commit, "../Py/const.py", "6pt")

  # footer, output and compile
  txt_out += footer()
  f = io.open("./temp.tex", "w")
  f.write(txt_out)
  f.close()
  os.system("make")
  os.system("mkdir -p ./archive")
  os.system("cp -f ./out.pdf ./archive/%s_%s_%dsteps.pdf" % (datestr, last_commit, const.STEPS))
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

def figure_table(expname, file_list, nx, ny, date, commit):
  if ny == 1 and nx == 1:
    width_cell = "0.96"
    width_img = "90mm"
  elif ny == 3 and nx == 3:
    width_cell = "0.32"
    width_img = "36mm"
  elif ny == 4 and nx == 4:
    width_cell = "0.24"
    width_img = "27mm"
  else:
    print("figure_table skipped due to unrecognized nx and ny")
    return ""

  content = """
    \\begin{frame}
    \\frametitle{@@expname@@}
    \\vspace*{-10mm}
    \\begin{figure}[h]
      \\flushleft
  """[1:-1]
  content = content.replace('@@expname@@', date + " " + commit + " "  + expname.replace("_", "\\_"))

  for iy in range(ny):
    for ix in range(nx):
      tex_figure = """
        \\begin{minipage}[b]{@@widthcell@@\\linewidth}
          \\centering
          \\includegraphicsmaybe[width=@@widthimg@@]{"@@filename@@"}
          % \\subcaption{@@num@@}
        \\end{minipage}
        """[1:-1]
      tex_figure = tex_figure.replace('@@filename@@', file_list[ix + iy * nx])
      tex_figure = tex_figure.replace('@@expname@@', expname)
      tex_figure = tex_figure.replace('@@widthcell@@', width_cell)
      tex_figure = tex_figure.replace('@@widthimg@@', width_img)
      content += tex_figure
    content += "\\\\"

  closing = """
    % \\caption{polygons}\\label{reg_poly}
    \\end{figure}
    \\end{frame}
  """[1:-1]
  content += closing

  return textwrap.dedent(content[1:-1])

def write_txt(date, commit, path, size):
  if not os.path.isfile(path):
    return ""

  content = """
    \\begin{frame}[allowframebreaks]
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

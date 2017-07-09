#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, textwrap, datetime, sys, subprocess
sys.path.append("../Py")
import const

def main():
  git_output = str(subprocess.check_output("git show HEAD | head -n1", shell=True))
  last_commit = git_output.split()[1][:7]
  date = datetime.datetime.now()
  datestr = "%04d%02d%02d" % (date.year, date.month, date.day)
  txt_out = header(datestr, last_commit)
  expname = os.environ.get("BATCH_JOB_NAME", "")
  commit_name = last_commit + " " + expname

  # rmse and lyapunov exponents
  txt_out += figure_table("", ["../image/true/rmse_bar.pdf"], 1, 1, datestr, commit_name)
  txt_out += write_txt(datestr, commit_name,
                        "../image/true/rmse.txt", "9pt", "rmse")
  txt_out += write_txt(datestr, commit_name,
                        "../data/lyapunov.txt", "5.4pt", "lyapunov exponents")

  # rmse-spread comparison
  for strx in ["extra", "trop", "ocean"]:
    filelist = []
    for exp in const.EXPLIST:
      filelist.append("../image/%s/%s_%s_time.pdf" % (exp["name"], exp["name"], strx))
    if len(filelist) <= 9:
      txt_out += figure_table(strx, filelist, 3, 3, datestr, commit_name)
    else:
      txt_out += figure_table(strx, filelist, 4, 4, datestr, commit_name)

  # all figures
  for exp in const.EXPLIST:
    filelist = []
    imglist = ["anl_covar_logrms", "back_covar_logrms", "anl_covar_mean", "back_covar_mean", \
               "extra_traj", "trop_traj", "ocean_traj", "back_cosine_mean", \
               "extra_val", "trop_val", "ocean_val", "anl_cosine_mean", \
               "extra_time", "trop_time", "ocean_time", ""]
    for imgname in imglist:
      filelist.append("../image/%s/%s_%s.pdf" % (exp["name"], exp["name"], imgname))
    txt_out += figure_table(exp["name"], filelist, 4, 4, datestr, commit_name)

  # conditions
  txt_out += write_txt(datestr, commit_name, "../Py/const.py", "6pt", "settings")
  md_raw = subprocess.getoutput('md5sum ../data/*').split("\n")
  md_raw.sort()
  md_txt = ""
  for line in md_raw:
    if ("cycle" in line) or ("/true.bin" in line) or ("/obs.bin" in line):
      hash = line.split()[0]
      filename = line.split()[1]
      md_txt += hash[0:4] + " " + filename[8:] + "\n"
  f = io.open("./md5sum.txt", "w")
  f.write(md_txt)
  f.close()

  txt_out += write_txt(datestr, commit_name, "./md5sum.txt", "9pt", "checksum")

  # footer, output and compile
  txt_out += footer()
  f = io.open("./temp.tex", "w")
  f.write(txt_out)
  f.close()
  os.system("make")
  os.system("mkdir -p ./archive")
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
    \\setbeamerfont{frametitle}{size=\\small}

    \\newcommand{\\includegraphicsmaybe}[2][width=10mm]{
      \\IfFileExists{#2}{\\includegraphics[#1]{#2}}{\\includegraphics[#1]{dummy.png}
    }}


    \\begin{document}
    % \\maketitle
  """
  header = header.replace('@@date@@', sanitize(date))
  header = header.replace('@@title@@', sanitize(last_commit))
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

  while len(file_list) < (nx * ny):
    file_list.append("")

  content = """
    \\begin{frame}
    \\frametitle{@@expname@@}
    \\vspace*{-10mm}
    \\begin{figure}[h]
      \\flushleft
  """[1:-1]
  content = content.replace('@@expname@@', sanitize(date + " " + commit + " "  + expname))

  for iy in range(ny):
    for ix in range(nx):
      tex_figure = """
        \\begin{minipage}[b]{@@widthcell@@\\linewidth}
          \\centering
          \\includegraphicsmaybe[width=@@widthimg@@]{@@filename@@}
        \\end{minipage}
        """[1:-1]
      tex_figure = tex_figure.replace('@@filename@@', file_list[ix + iy * nx])
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

def write_txt(date, commit, path, size, title):
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
  content = content.replace('@@expname@@', sanitize(date + " " + commit + " " + title))
  content = content.replace('@@filepath@@', path)
  content = content.replace('@@size@@', size)

  return textwrap.dedent(content[1:-1])

def footer():
  footer = """
    \\end{document}
  """
  return textwrap.dedent(footer[1:-1])

def sanitize(txt):
  txt = txt.replace('\\', '\\textbackslash{}')
  txt = txt.replace('{', '\\{')
  txt = txt.replace('}', '\\}')
  txt = txt.replace('$', '\\$')
  txt = txt.replace('[', '［')
  txt = txt.replace(']', '］')
  txt = txt.replace('&', '\\&')
  txt = txt.replace('#', '\\#')
  txt = txt.replace('%', '\\%')
  txt = txt.replace('^', '\\textasciicircum{}')
  txt = txt.replace('_', '\\_')
  txt = txt.replace('~', '\\textasciitilde{}')
  txt = txt.replace('>', '\\textless{}')
  txt = txt.replace('<', '\\textgreater{}')
  txt = txt.replace('|', '\\textbar{}')
  txt = txt.replace('"', '\\textquotedbl{}')
  txt = txt.replace("'", '\\textquotesingle{}')
  txt = txt.replace('`', '\\textasciigrave{}')
  return txt

main()

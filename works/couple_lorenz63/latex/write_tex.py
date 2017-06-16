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
  # explist = const.EXPLIST
  explist = ["etkf", "tdvar", "fdvar"]
  filebase = "../image/@@expname@@_@@yname@@_int8/@@expname@@_@@yname@@_int8_@@xname@@_time.png"
  for expname in explist:
    xlist = ["extro", "trop", "ocn"]
    ylist = ["strong", "weak", "non"]
    txt_out += figure_table(expname, filebase, xlist, ylist, datestr, last_commit)
  txt_out += write_rmse(datestr, last_commit)
  txt_out += footer()

  f = io.open("./temp.tex", "w")
  f.write(txt_out)
  f.close()
  os.system("make")
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
  content = content.replace('@@expname@@', date + " " + commit + " " + expname)

  for yname in ylist:
    for xname in xlist:
      tex_figure = """
        \\begin{minipage}[b]{0.32\\linewidth}
          \\centering
          \\includegraphics[width=36mm]{@@filebase@@}
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

def write_rmse(date, commit):
  content = """
    \\begin{frame}
    \\frametitle{@@expname@@}
    \\verbatiminput{../image/true/rmse.txt}
    \\end{frame}
  """[1:-1]
  content = content.replace('@@expname@@', date + " " + commit + " rmse")

  return textwrap.dedent(content[1:-1])

def footer():
  footer = """
    \\end{document}
  """
  return textwrap.dedent(footer[1:-1])

main()

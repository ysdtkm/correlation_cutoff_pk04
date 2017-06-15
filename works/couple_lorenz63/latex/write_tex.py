#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, textwrap, datetime

def main():
  txt_out = header()
  txt_out = txt_out + figure_table()
  txt_out = txt_out + footer()
  f = io.open("./temp.tex", "w")
  f.write(txt_out)
  f.close()
  os.system("make")
  return 0

def header():
  date = datetime.datetime.now()
  header = """
    \\documentclass{beamer}
    \\usepackage{xltxtra}
    \\usepackage{amsmath}
    \\usepackage[subrefformat=parens]{subcaption}
    \\usetheme{Boadilla}
    \\usecolortheme{beaver}
    \\title{Coupled Lorenz63 model experiments}
    \\date{@@date@@}
    \\author{Takuma Yoshida}
    \\parskip=12pt
    \\parindent=0pt

    %gets rid of bottom navigation bars, navigation symbols and footer
    \\setbeamertemplate{footline}[frame number]{}
    \\setbeamertemplate{navigation symbols}{}
    \\setbeamertemplate{footline}{}

    \\begin{document}
    \\maketitle
  """
  header = header.replace('@@date@@', "%04d%02d%02d" % (date.year, date.month, date.day))
  return textwrap.dedent(header[1:-1])

def figure_table():
  content = """
    \\begin{frame}
    \\frametitle{figures}
    \\begin{figure}[h]
      \\flushleft
      \\begin{minipage}[b]{0.32\\linewidth}
        \\centering
        \\includegraphics[keepaspectratio, scale=0.23]{../image/etkf_weak_int8/etkf_weak_int8_trop_time.png}
        % \\subcaption{trop}\\label{poly04}
      \\end{minipage}
      \\begin{minipage}[b]{0.32\\linewidth}
        \\centering
        \\includegraphics[keepaspectratio, scale=0.23]{../image/etkf_weak_int8/etkf_weak_int8_trop_time.png}
        % \\subcaption{trop}\\label{poly06}
      \\end{minipage}
      \\begin{minipage}[b]{0.32\\linewidth}
        \\centering
        \\includegraphics[keepaspectratio, scale=0.23]{../image/etkf_weak_int8/etkf_weak_int8_trop_time.png}
        % \\subcaption{trop}\\label{poly08}
      \\end{minipage} \\\\

      \\begin{minipage}[b]{0.32\\linewidth}
        \\centering
        \\includegraphics[keepaspectratio, scale=0.23]{../image/etkf_weak_int8/etkf_weak_int8_trop_time.png}
        % \\subcaption{trop}\\label{poly12}
      \\end{minipage}
      \\begin{minipage}[b]{0.32\\linewidth}
        \\centering
        \\includegraphics[keepaspectratio, scale=0.23]{../image/etkf_weak_int8/etkf_weak_int8_trop_time.png}
        % \\subcaption{trop}\\label{poly20}
      \\end{minipage}
      % \\caption{polygons}\\label{reg_poly}
    \\end{figure}
    \\end{frame}
  """
  return textwrap.dedent(content[1:-1])

def footer():
  footer = """
    \\end{document}
  """
  return textwrap.dedent(footer[1:-1])

main()

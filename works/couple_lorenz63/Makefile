#================================
# [Usage]
# .make/以下に作成する空ファイルのタイムスタンプで
# 各ステップの最終実行時刻を評価
#
# STEP_ALL : 全ステップ名。数字以外も使用可能
# src_step : 実行判断に利用されるソースファイル
# dep_step : 依存するステップ名
# out_step : 出力ファイル。make cleanで消去される。ワイルドカード可
# cmd_step : 実行コマンド
#================================

STEP_ALL = calc plot tex save
# time = time --format="spent %E sec"
time =

src_calc = Py/main.py Py/const.py Py/model.py Py/etkf.py Py/fdvar.py Py/tdvar.py \
           Py/vectors.py
dep_calc =
out_calc = data/*.bin
cmd_calc = rm -rf data; mkdir -p data; $(time) python3 Py/main.py

src_plot = Py/plot.py Py/const.py
dep_plot = calc
out_plot = image/*
cmd_plot = rm -rf image; mkdir -p image; $(time) python3 Py/plot.py

src_tex = latex/write_tex.py
dep_tex = plot
out_tex =
cmd_tex = cd latex && python3 write_tex.py

src_save =
dep_save = tex
out_save = raw tar
cmd_save = rm -rf raw tar && mkdir -p raw tar && mv -f data image tar/ && mv latex/out.pdf raw

# ===============================================
# end of settings
# ===============================================

VPATH = .make
all: $(STEP_ALL)

define rule_comm
$1: $(src_$1) $(dep_$1)
	@echo ""
	@echo "STEP $1:"
	$(cmd_$1)
	@mkdir -p $(VPATH)
	@touch $(VPATH)/$1
clean_$1:
	@echo "clean $1:"
	rm -rf $(out_$1)
	rm -f $(VPATH)/$1
endef

# 第三要素の$i部分をSTEP_ALL変数の各要素で置換
$(foreach i, $(STEP_ALL), $(eval $(call rule_comm,$i)))

clean:
	rm -rf $(foreach i,$(STEP_ALL),$(out_$i))
	@rm -rf $(VPATH)

.PHONY: all clean

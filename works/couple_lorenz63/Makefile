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

STEP_ALL = 1

src_1 = Py/control/main.py Py/control/const.py
dep_1 =
out_1 = data/*.bin
cmd_1 = python Py/control/main.py

src_2 = Py/plot.py Py/const.py
dep_2 = 1
out_2 = image/*
cmd_2 = python Py/plot.py

# ===============================================
# end of settings
# ===============================================

VPATH = .make
all: $(STEP_ALL)

define rule_comm
$1: $(src_$1) $(dep_$1)
	@echo "STEP $1:"
	$(cmd_$1)
	@mkdir -p $(VPATH)
	@touch $(VPATH)/$1
endef

# 第三要素の$i部分をSTEP_ALL変数の各要素で置換
$(foreach i, $(STEP_ALL), $(eval $(call rule_comm,$i)))

clean:
	rm -rf $(foreach i,$(STEP_ALL),$(out_$i))
	@rm -rf $(VPATH)

.PHONY: all clean

########################################################################
# Configuration
OUTPUT_DIR = output
RESULT_DIR = results
PYTHON = python3
GNUPLOT = gnuplot
GHOSTSCRIPT = gs
GHOSTSCRIPT_OPTIONS = -sDEVICE=pdfwrite -dColorConversionStrategy=/sRGB -dProcessColorModel=/DeviceRGB -dCompatibilityLevel=1.5


########################################################################
# Dependencies
all: \
  $(RESULT_DIR)/tab-comparison.tex \
  $(RESULT_DIR)/tab-comparison-attribute.tex \
  $(RESULT_DIR)/tab-comparison-size.tex \
  $(RESULT_DIR)/tab-comparison-time.tex \
  $(RESULT_DIR)/plot-accuracy.pdf \
  $(RESULT_DIR)/plot-balanced-accuracy.pdf \
  $(RESULT_DIR)/plot-fairness.pdf \
  $(RESULT_DIR)/plot-time.pdf \
  $(RESULT_DIR)/plot-size.pdf

$(RESULT_DIR)/raw.json: \
  bin/silva bin/meta-silvae \
  adult/dataset.csv compas/dataset.csv crime/dataset.csv german/dataset.csv health/dataset.csv

$(RESULT_DIR)/tab-comparison.tex: $(RESULT_DIR)/raw.json
$(RESULT_DIR)/tab-comparison-attribute.tex: $(RESULT_DIR)/raw.json
$(RESULT_DIR)/tab-comparison-size.tex: $(RESULT_DIR)/raw.json
$(RESULT_DIR)/tab-comparison-time.tex: $(RESULT_DIR)/raw.json

$(RESULT_DIR)/plot-metrics.csv: $(RESULT_DIR)/raw.json
$(RESULT_DIR)/plot-accuracy.pdf: $(RESULT_DIR)/plot-metrics.csv
$(RESULT_DIR)/plot-balanced-accuracy.pdf: $(RESULT_DIR)/plot-metrics.csv
$(RESULT_DIR)/plot-fairness.pdf: $(RESULT_DIR)/plot-metrics.csv
$(RESULT_DIR)/plot-time.pdf: $(RESULT_DIR)/plot-metrics.csv
$(RESULT_DIR)/plot-size.pdf: $(RESULT_DIR)/plot-metrics.csv

.PHONY: clean


########################################################################
# Receipes
%/dataset.csv: %/get-dataset.py
	@echo "Preprocessing $@..."
	@python3 $^

bin/silva:
	@echo "Installing silva from GitHub..."
	@make -C silva/src install
	@mv silva/bin/silva bin/silva

bin/meta-silvae:
	echo "Installing meta-silvae from GitHub..."
	@make -C meta-silvae/src install
	@mv meta-silvae/bin/meta-silvae bin/meta-silvae

$(RESULT_DIR)/raw.json:
	@echo "Running tests..."
	@mkdir -p `dirname $@`
	@mkdir -p $(OUTPUT_DIR)
	@$(PYTHON) run-tests.py $@

$(RESULT_DIR)/tab-comparison.tex:
	@echo "Generating fairness comparison table..."
	@$(PYTHON) process-output.py $< $(RESULT_DIR)

$(RESULT_DIR)/tab-comparison-attribute.tex:
	@echo "Generating attribute fairness comparison table..."
	@$(PYTHON) process-output.py $< $(RESULT_DIR)

$(RESULT_DIR)/tab-comparison-size.tex:
	@echo "Generating model size comparison table..."
	@$(PYTHON) process-output.py $< $(RESULT_DIR)

$(RESULT_DIR)/tab-comparison-time.tex:
	@echo "Generating fairness verification time comparison table..."
	@$(PYTHON) process-output.py $< $(RESULT_DIR)

$(RESULT_DIR)/plot-metrics.csv:
	@echo "Preparing plot data..."
	@$(PYTHON) process-output.py $< $(RESULT_DIR)

$(RESULT_DIR)/plot-accuracy.pdf:
	@echo "Plotting accuracy..."
	@$(GNUPLOT) plot-metrics.gpl
	@$(GHOSTSCRIPT) -o $@2 $(GHOSTSCRIPT_OPTIONS) $@
	@mv $@2 $@

$(RESULT_DIR)/plot-balanced-accuracy.pdf:
	@echo "Plotting balanced accuracy..."
	@$(GNUPLOT) plot-metrics.gpl

$(RESULT_DIR)/plot-fairness.pdf:
	@echo "Plotting fairness..."
	@$(GNUPLOT) plot-metrics.gpl

$(RESULT_DIR)/plot-time.pdf:
	@echo "Plotting verification time..."
	@$(GNUPLOT) plot-metrics.gpl

$(RESULT_DIR)/plot-size.pdf:
	@echo "Plotting model size..."
	@$(GNUPLOT) plot-metrics.gpl

clean:
	@rm -fR __pycache__
	@rm -fR */*.csv
	@rm -fR */*.zip
	@rm -fR $(OUTPUT_DIR)
	@rm -fR $(RESULT_DIR)

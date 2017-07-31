import line_profiler, time
import main, fdvar, plot

pr = line_profiler.LineProfiler()
pr.add_function(plot.plot_rmse_spread)
pr.runcall(plot.plot_all)
pr.print_stats()


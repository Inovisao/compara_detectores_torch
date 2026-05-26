library("ggplot2")
library("gridExtra")
library("dplyr")

options(scipen = 999)

base_df <- read.table('../results/results_base.csv', sep = ',', header = TRUE)
ft_df   <- read.table('../results/results_finetune.csv', sep = ',', header = TRUE)

base_df$config <- "Base"
ft_df$config   <- "Finetune"

dados <- rbind(base_df, ft_df)
dados$config <- factor(dados$config, levels = c("Base", "Finetune"))

metricas <- c("mAP50", "mAP75", "mAP", "precision", "recall", "fscore", "MAE", "RMSE", "r")
graficos <- list()

for (i in seq_along(metricas)) {
  metrica <- metricas[[i]]
  g <- ggplot(dados, aes_string(x = "config", y = metrica, fill = "config")) +
    geom_boxplot() +
    scale_fill_brewer(palette = "Set2") +
    labs(title = sprintf("Boxplot — %s", metrica), x = "Config", y = metrica) +
    theme(legend.position = "none", plot.title = element_text(hjust = 0.5))
  graficos[[i]] <- g
}

g <- grid.arrange(grobs = graficos, ncol = 3)
ggsave("../results/boxplot_compare.png", g, width = 14, height = 10)

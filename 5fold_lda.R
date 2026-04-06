library(UBL)
library(MASS)
library(caret)

# load the original CSV file (adjust as needed)
original_csv <- "/Users/zshane/Documents/um_msc/efd_h10/h10_dm_final_norm.csv"

cell_name <- "dm"

# output folder (adjust as needed)
output_folder <- "/Users/zshane/Documents/um_msc/revised_efd_h10"

if (!dir.exists(output_folder)) {
  dir.create(output_folder)
}

# Read the original CSV file
original_data <- read.csv(original_csv, header = TRUE, stringsAsFactors = FALSE)

## Get the IDs for k-folds
set.seed(123)
folds <- createFolds(original_data$species, k = 5, list = TRUE, returnTrain = FALSE)

# Loop through all 5 folds
for (fold in 1:5) {
  # Get train and test sets for this fold
  test_idx <- folds[[fold]]
  train_fd <- original_data[-test_idx, ]
  test_fd <- original_data[test_idx, ]
  
  cat("\nFold", fold, "mapping summary:")
  cat("\nNumber of matched training samples:", nrow(train_fd))
  cat("\nNumber of matched testing samples:", nrow(test_fd), "\n")

  # -----------------------------
  # Preprocess Training and Testing Data
  # -----------------------------
  
  # Convert species column to factor
  train_fd$species <- as.factor(train_fd$species)
  test_fd$species <- as.factor(test_fd$species)
  
  # Extract predictor columns and convert to numeric
  train_predictors <- as.data.frame(lapply(train_fd[, 4:ncol(train_fd)], as.numeric))
  test_predictors <- as.data.frame(lapply(test_fd[, 4:ncol(test_fd)], as.numeric))
  
  # Rebuild clean data frames
  train_fd <- data.frame(
    sample_name = train_fd$name,
    species = train_fd$species,
    train_predictors,
    stringsAsFactors = FALSE
  )
  
  test_fd <- data.frame(
    sample_name = test_fd$name,
    species = test_fd$species,
    test_predictors,
    stringsAsFactors = FALSE
  )
  
  # -----------------------------
  # Remove Constant Columns
  # -----------------------------
  # Explicitly remove a1, b1, c1 as requested
  cols_to_remove <- c("a1", "b1", "c1")
  
  # Check which ones actually exist in the data
  existing_cols <- intersect(cols_to_remove, colnames(train_fd))
  
  if (length(existing_cols) > 0) {
    cat("\nRemoving specified constant columns:", paste(existing_cols, collapse=", "), "\n")
    train_fd <- train_fd[, !colnames(train_fd) %in% existing_cols]
    test_fd <- test_fd[, !colnames(test_fd) %in% existing_cols]
  }

  # -----------------------------
  # Perform LDA on the Training and Testing Sets
  # -----------------------------
  
  # Debug prints before LDA
  cat("\nFold", fold, "species summary:")
  cat("\nNumber of unique species in training:", length(unique(train_fd$species)))
  print(table(train_fd$species))
  
  # Fit LDA
  model <- lda(species ~ ., data = train_fd[, -1])

  # Save LDA model info to a text file
  model_info_file <- file.path(output_folder, paste0("lda_modelinfo_fold_", fold, "_", cell_name, ".txt"))
  capture.output(print(model), file = model_info_file)
  cat("Saved LDA model info for fold", fold, "to:", model_info_file, "\n")

  # Check LDA output
  cat("\nNumber of LD columns:", ncol(model$scaling))
  
  # Get the transformation (coefficients)
  coef <- model$scaling
  loadings_df <- as.data.frame(coef)
  
  # Add row names (variable names)
  rownames(loadings_df) <- colnames(train_fd[, -c(1,2)])
  
  # Create loadings output file name
  loadings_file <- file.path(output_folder, paste0("lda_loadings_fold_", fold, "_",cell_name,".txt"))
  
  # Write loadings to file
  write.table(loadings_df,
              file = loadings_file,
              sep = "\t",
              quote = FALSE,
              col.names = TRUE,
              row.names = TRUE)
              
  cat("\nSaved loadings for fold", fold, "to:", loadings_file, "\n")

  
  # Compute LDA projections.
  # For predictors, exclude both sample_name and species:
  ld_train <- as.matrix(train_fd[, -c(1,2)]) %*% coef
  ld_test  <- as.matrix(test_fd[, -c(1,2)]) %*% coef
  
  # Center the training LDA components (subtract their mean).
  mean_ld_train <- colMeans(ld_train)
  saveRDS(mean_ld_train, file.path(output_folder,paste0("mean_ld_train_fd_", fold, "_",cell_name,".rds")))
  ld_train_centered <- sweep(ld_train, 2, mean_ld_train, FUN = "-")
  
  # Center the testing LDA components using the training-set means.
  ld_test_centered <- sweep(ld_test, 2, mean_ld_train, FUN = "-")
  
  # Build data frame for training
  # Keep a row_id so original sample names can be restored after ENN
  df_train <- data.frame(
    row_id = seq_len(nrow(train_fd)),
    species = train_fd$species,
    ld_train_centered,
    stringsAsFactors = FALSE
  )

  # Store original sample-name lookup
  name_lookup <- setNames(train_fd$sample_name, df_train$row_id)

  # Output df_train (after LDA, before SMOTE) for cross-checking
  pre_smote_file <- file.path(output_folder, paste0("mh_fold_", fold, "_pre_smote_train.csv"))
  write.csv(df_train, file = pre_smote_file, row.names = FALSE)

  # -----------------------------
  # SMOTE + ENN
  # -----------------------------
  set.seed(123)

  class_sizes <- table(df_train$species)
  target_size <- 41

  # Oversample only classes below target_size
  C.perc_list <- lapply(names(class_sizes), function(sp) {
    size <- as.integer(class_sizes[[sp]])
    if (size < target_size) target_size / size else 1
  })
  names(C.perc_list) <- names(class_sizes)

  df_train_smote <- SmoteClassif(
    species ~ . - row_id,
    df_train,
    C.perc = C.perc_list
  )

  # ENN cleaning
  df_train_enn <- ENNClassif(
    species ~ . - row_id,
    df_train_smote
  )[[1]]

  # ENN does not guarantee a fixed final size, so cap each class at 41
  df_train_enn <- do.call(rbind, lapply(split(df_train_enn, df_train_enn$species), function(d) {
    if (nrow(d) > target_size) {
      d[sample(seq_len(nrow(d)), target_size), , drop = FALSE]
    } else {
      d
    }
  }))
  rownames(df_train_enn) <- NULL

  # Restore / generate sample names
  df_train_enn$sample_name <- NA_character_

  orig_idx <- which(!is.na(df_train_enn$row_id))
  df_train_enn$sample_name[orig_idx] <- name_lookup[as.character(df_train_enn$row_id[orig_idx])]

  syn_idx <- which(is.na(df_train_enn$row_id))
  if (length(syn_idx) > 0) {
    syn_sp <- as.character(df_train_enn$species[syn_idx])
    syn_num <- ave(syn_idx, syn_sp, FUN = seq_along)
    df_train_enn$sample_name[syn_idx] <- paste0(
      "synthetic_",
      gsub("[^A-Za-z0-9]", "", syn_sp),
      "_",
      syn_num
    )
  }

  df_train_enn$weight <- 1
  df_train_final <- df_train_enn[, c("sample_name", "species", colnames(ld_train_centered), "weight")]

  # Restore / generate sample names
  df_train_enn$sample_name <- NA_character_

  # Original samples: restore names by row_id
  orig_idx <- which(!is.na(df_train_enn$row_id))
  df_train_enn$sample_name[orig_idx] <- name_lookup[as.character(df_train_enn$row_id[orig_idx])]

  # Synthetic samples: generate names per species
  syn_idx <- which(is.na(df_train_enn$row_id))
  if (length(syn_idx) > 0) {
    syn_sp <- as.character(df_train_enn$species[syn_idx])
    syn_num <- ave(syn_idx, syn_sp, FUN = seq_along)
    df_train_enn$sample_name[syn_idx] <- paste0(
      "synthetic_",
      gsub("[^A-Za-z0-9]", "", syn_sp),
      "_",
      syn_num
    )
  }

  df_train_enn$weight <- 1

  # Final training set
  df_train_final <- df_train_enn[, c("sample_name", "species", colnames(ld_train_centered), "weight")]
  
  # Prepare testing set
  df_test <- data.frame(
    sample_name = test_fd$sample_name,
    species = test_fd$species,
    ld_test_centered,
    stringsAsFactors = FALSE
  )
  df_test$weight <- 0
  
  class_sizes_final <- table(df_train_final$species)
  cat("\nFinal species counts in training set after SMOTE:\n")
  print(class_sizes_final)

  cat("\nNumber of unique species in Testing:", length(unique(test_fd$species)))
  print(table(test_fd$species))

  # Combine final training data (original + synthetic) with test data
  combined_ld <- rbind(df_train_final, df_test)

  # Define the family mapping as a named vector
  family_map <- c(
    "C.vicina" = "Calliphoridae",
    "Ch.albiceps" = "Calliphoridae",
    "Ch.albiceps_mutant" = "Calliphoridae",
    "Ch.bezziana" = "Calliphoridae",
    "Ch.megacephala" = "Calliphoridae",
    "Ch.nigripes" = "Calliphoridae",
    "Ch.rufifacies" = "Calliphoridae",
    "L.sericata" = "Calliphoridae",
    "A.gressitti" = "Sarcophagidae",
    "B.karnyi" = "Sarcophagidae",
    "Le.alba" = "Sarcophagidae",
    "Z.aquila" = "Sarcophagidae",
    "S.princeps" = "Sarcophagidae",
    "Sy.nudiseta" = "Muscidae"
  )

  # Add the family column as the third column in combined_ld
  combined_ld$family <- family_map[as.character(combined_ld$species)]
  combined_ld <- combined_ld[, c("sample_name", "species", "family", setdiff(names(combined_ld), c("sample_name", "species", "family")))]
    
  # -----------------------------
  # Define the Output File Name
  # -----------------------------
  # Construct the output file name: e.g., "fold1_guide_in.txt"
  output_file <- file.path(output_folder, paste0("fd", fold, "_",cell_name,"_guide_data.txt"))
  
  # Write the result (with header) to a comma-separated file.
  write.table(combined_ld,
              file = output_file,
              sep = " ",
              quote = TRUE,
              row.names = FALSE,
              col.names = TRUE)
  
  cat("Saved file:", output_file, "\n")
}

cat("Processing completed.\n")

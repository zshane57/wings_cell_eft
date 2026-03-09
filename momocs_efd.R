# Load necessary libraries
library(imager)
library(Momocs)
library(foreach)
library(doParallel)

# Define the parent directory containing species folders
parent_folder <- "/Users/zshane/Documents/um_msc/masks"  # Update this path
output_path <- "/Users/zshane/Documents/um_msc/efd_h10"  # Update this path

# Set up parallel backend
num_cores <- parallel::detectCores() - 1  # Use all available cores minus one
cl <- makeCluster(num_cores)
registerDoParallel(cl)
cat("Using", num_cores, "cores for parallel processing.\n")

harmonics <- 10  # Define the harmonics to compute

# Get all species folders
species_folders <- list.dirs(path = parent_folder, full.names = TRUE, recursive = FALSE)

# Define family groups
family_groups <- list(
    Calliphoridae = c("C.vicina", "Ch.albiceps", "Ch.albiceps_mutant", "Ch.bezziana", 
                      "Ch.megacephala", "Ch.nigripes", "Ch.rufifacies", "L.sericata"),
    Sarcophagidae = c("A.gressitti", "B.karnyi", "Le.alba", "Z.aquila", "S.princeps"),
    Muscidae = c("Sy.nudiseta")
)

# Helper function to get family from species
get_family <- function(species) {
    for (fam in names(family_groups)) {
        if (species %in% family_groups[[fam]]) return(fam)
    }
    return(NA)
}

# Initialize a list to collect all results across species
all_results <- list()

# Loop through each species folder
for (species_folder in species_folders) {
    species_name <- basename(species_folder)
    cat("Processing species:", species_name, "\n")
    start_time <- Sys.time()
    
    # Only get the "mask5" folder within the species folder
    mask5_folder <- file.path(species_folder, "masks9")
    if (!dir.exists(mask5_folder)) {
        cat("  No mask5 folder found in:", species_folder, "\n")
        next
    }
    mask_folders <- mask5_folder

    # Loop through each mask folder (now only mask5)
    for (mask_folder in mask_folders) {
        cat("  Processing mask folder:", mask_folder, "\n")
        mask_files <- list.files(path = mask_folder, pattern = "*.png", full.names = TRUE)


        results <- foreach(mask_file = mask_files, .combine = rbind, .packages = c("imager", "Momocs")) %dopar% {
            cat("    Processing file:", mask_file, "\n")
            img <- load.image(mask_file)
            image <- mirror(img, "y")
            contour_coords <- imager::contours(image)
            if (!is.null(contour_coords) && length(contour_coords) > 0) {
                contour_areas <- sapply(contour_coords, function(contour) {
                    x <- contour$x
                    y <- contour$y
                    abs(sum(x[-1] * y[-length(y)] - x[-length(x)] * y[-1]) / 2)
                })
                largest_contour <- contour_coords[[which.max(contour_areas)]]
                contour_matrix <- cbind(largest_contour$x, largest_contour$y)
                if (nrow(contour_matrix) > 5) {
                    nb.h <- harmonics #min(harmonics, nrow(contour_matrix) %/% 2)
                    tryCatch({
                        coe <- efourier(contour_matrix, nb.h = nb.h, norm = FALSE)
                        actual_harmonics <- length(coe$an)
                        coef_data <- data.frame(
                            t(c(coe$an, coe$bn, coe$cn, coe$dn)),
                            name = paste0(sub("_cleanup.*", "", tools::file_path_sans_ext(basename(mask_file))), ".png"),
                            species = species_name,
                            family = get_family(species_name)
                        )
                        colnames(coef_data)[1:(4 * actual_harmonics)] <- unlist(lapply(c("a", "b", "c", "d"), function(prefix) paste0(prefix, 1:actual_harmonics)))
                        return(coef_data)
                    }, error = function(e) {
                        cat("      Error in efourier for:", mask_file, "-", e$message, "\n")
                        return(NULL)
                    })
                } else {
                    cat("      Insufficient points in largest contour for:", mask_file, "\n")
                    return(NULL)
                }
            } else {
                cat("      Failed to extract contours for:", mask_file, "\n")
                return(NULL)
            }
        }
        
        # Append results for this mask folder to all_results
        all_results[[paste0(species_name, "_", basename(mask_folder))]] <- results
    }
    end_time <- Sys.time()
    elapsed_time <- end_time - start_time
    cat("Processing time for species", species_name, ":", elapsed_time, "seconds\n")
}

# Combine all results into a single dataframe
combined_all_results <- do.call(rbind, all_results)

# Ensure the columns "name", "species", and "family" are the first three columns in the final dataframe
if (all(c("name", "species", "family") %in% colnames(combined_all_results))) {
    combined_all_results <- combined_all_results[, c("name", "species", "family", setdiff(colnames(combined_all_results), c("name", "species", "family")))]
}

# Define the output CSV file path for all coefficients
output_csv <- file.path(output_path, "h10_efd_coef_dm.csv")
write.csv(combined_all_results, file = output_csv, row.names = FALSE)
cat("All EFD coefficients saved to", output_csv, "\n")

# Stop and clean up the parallel cluster
stopCluster(cl)
cat("Parallel processing completed.\n")
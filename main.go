package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"github.com/spf13/cobra"
)

func getPythonCommand() string {
	switch runtime.GOOS {
	case "windows":
		return "python" // Para Windows
	default:
		return "python3" // Para macOS e Linux
	}
}
func runPythonScript(command string, args []string) {
	cmd := exec.Command(getPythonCommand(), append([]string{"autotrain.py", command}, args...)...)

	// Captura a saída do script Python
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("Erro ao executar o script Python: %s\n", err)
		fmt.Printf("Saída:\n%s\n", string(output))
		return
	}

	// Exibe a saída do script Python
	fmt.Printf("\n%s\n", string(output))
}

func main(){
	// base command
	var rootCmd = &cobra.Command{
		Use: "autotrain",
		Short: "A CLI to automate model training tasks",
	}

	// format
	var formatCmd = &cobra.Command{
		Use: "format",
		Short: "Formats the dataset for different tasks (object detection, segmentation)",
	}

	// generate dataset on segmentation format
	var segmentationCmd = &cobra.Command{
		Use:   "segmentation",
		Short: "Formats the dataset for segmentation",
		Run: func(cmd *cobra.Command, args []string) {
			// Captura os valores das flags
			inputDir, _ := cmd.Flags().GetString("input_dir")
			outputPositiveDir, _ := cmd.Flags().GetString("output_positive_dir")
			outputNegativeDir, _ := cmd.Flags().GetString("output_negative_dir")
			retrainModel, _ := cmd.Flags().GetString("model")
	
			// Constrói os argumentos para o script Python
			pythonArgs := []string{
				"--input_dir", inputDir,
				"--output_positive_dir", outputPositiveDir,
				"--output_negative_dir", outputNegativeDir,
				"--model", retrainModel,
			}
	
			// Chama a função para executar o script Python
			runPythonScript("format", append([]string{"segmentation"}, pythonArgs...))
		},
	}

	// generate dataset on object detection format
	var objectDetectionCmd = &cobra.Command{
		Use:   "object_detection",
		Short: "Formats the dataset for object detection",
		Run: func(cmd *cobra.Command, args []string) {
			// Captura os valores das flags
			inputDir, _ := cmd.Flags().GetString("input_dir")
			outputPositiveDir, _ := cmd.Flags().GetString("output_positive_dir")
			outputNegativeDir, _ := cmd.Flags().GetString("output_negative_dir")
			retrainModel, _ := cmd.Flags().GetString("model")
	
			// Constrói os argumentos para o script Python
			pythonArgs := []string{
				"--input_dir", inputDir,
				"--output_positive_dir", outputPositiveDir,
				"--output_negative_dir", outputNegativeDir,
				"--model", retrainModel,
			}
	
			// Chama a função para executar o script Python
			runPythonScript("format", append([]string{"object_detection"}, pythonArgs...))
		},
	}

	// split dataset into 70% for train and 30% for validation
	var splitDatasetCmd = &cobra.Command{
		Use:   "split_dataset",
		Short: "Divide o dataset em treino (70%) e validação (30%)",
		Run: func(cmd *cobra.Command, args []string) {
			// Captura os valores das flags
			outputPositiveDir, _ := cmd.Flags().GetString("output_positive_dir")
			trainDir, _ := cmd.Flags().GetString("train_dir")
			valDir, _ := cmd.Flags().GetString("val_dir")
	
			// Construa os argumentos para passar para o script Python
			pythonArgs := []string{
				"--output_positive_dir", outputPositiveDir,
				"--train_dir", trainDir,
				"--val_dir", valDir,
			}
	
			// Executa o script Python com os argumentos
			runPythonScript("split_dataset", pythonArgs)
		},
	}

	// train the new model
	var trainCmd = &cobra.Command{
		Use:   "train",
		Short: "Treina o novo modelo",
		Run: func(cmd *cobra.Command, args []string) {
			// Obtenha os valores das flags e passe como argumentos para o script Python
			model, _ := cmd.Flags().GetString("model")
			datasetYaml, _ := cmd.Flags().GetString("dataset_yaml")
			device, _ := cmd.Flags().GetString("device")
			epochs, _ := cmd.Flags().GetInt("epochs")
			imgsz, _ := cmd.Flags().GetInt("imgsz")
	
			// Construa os argumentos para o script Python
			pythonArgs := []string{
				"--model", model,
				"--dataset_yaml", datasetYaml,
				"--device", device,
				"--epochs", fmt.Sprintf("%d", epochs),
				"--imgsz", fmt.Sprintf("%d", imgsz),
			}
	
			// Chama a função para executar o script Python
			runPythonScript("train", pythonArgs)
		},
	}

	segmentationCmd.Flags().String("input_dir", "", "Image directory")
	segmentationCmd.Flags().String("output_positive_dir", "", "Output positive image directory")
	segmentationCmd.Flags().String("output_negative_dir", "", "Output negative image directory")
	segmentationCmd.Flags().String("model", "", "Base model for classification")

	objectDetectionCmd.Flags().String("input_dir", "", "Image directory")
	objectDetectionCmd.Flags().String("output_positive_dir", "", "Output positive image directory")
	objectDetectionCmd.Flags().String("output_negative_dir", "", "Output negative image directory")
	objectDetectionCmd.Flags().String("model", "", "Base model for classification")

	trainCmd.Flags().String("model", "", "Location of the base model")
	trainCmd.Flags().String("dataset_yaml", "", "Location of .yaml for train/validation dataset")
	trainCmd.Flags().String("device", "cuda", "cuda or cpu")
	trainCmd.Flags().Int("epochs", 100, "Number of epochs")
	trainCmd.Flags().Int("imgsz", 640, "Image size")

	splitDatasetCmd.Flags().String("output_positive_dir", "", "Output positive image directory")
	splitDatasetCmd.Flags().String("train_dir", "train", "Directory for training images")
	splitDatasetCmd.Flags().String("val_dir", "val", "Directory for validation images")

	formatCmd.AddCommand(segmentationCmd)
	formatCmd.AddCommand(objectDetectionCmd)

	rootCmd.AddCommand(formatCmd)
	rootCmd.AddCommand(splitDatasetCmd)
	rootCmd.AddCommand(trainCmd)

	if err := rootCmd.Execute(); err != nil{
		fmt.Println(err)
		os.Exit(1)
	}
}
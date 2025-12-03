package main

import (
	"fmt"
	"os"

	"github.com/zamedic/voyageai"
)

func main() {
	vo := voyageai.NewClient(nil)

	embeddings, err := vo.Embed(
		[]string{
			"Embed this text please",
			"And this as well",
		},
		voyageai.ModelVoyage3Lite,
		nil,
	)

	if err != nil {
		fmt.Printf("Could not get embedding: %s", err.Error())
	}

	fmt.Printf("Embeddings (First 5): %v\n", embeddings.Data[0].Embedding[0:5])
	fmt.Printf("Usage: %v\n", embeddings.Usage)

	img, err := os.Open("./assets/gopher.png")
	if err != nil {
		fmt.Printf("Could not open image: %s", err.Error())
	}

	multimodalInput := []voyageai.MultimodalContent{
		{
			Content: []voyageai.MultimodalInput{
				{
					Type: "text",
					Text: "This is a picture of the Go mascot",
				},
				{
					Type:        "image_base64",
					ImageBase64: voyageai.MustGetBase64(img),
				},
			},
		},
	}

	mEmbedding, err := vo.MultimodalEmbed(multimodalInput, voyageai.ModelVoyageMultimodal3, nil)
	if err != nil {
		fmt.Printf("Could not get multimodal embedding: %s", err.Error())
	}
	fmt.Printf("Multimodal Embeddings (First 5): %v\n", mEmbedding.Data[0].Embedding[0:5])
	fmt.Printf("Usage: %v\n", mEmbedding.Usage)

	reranking, err := vo.Rerank(
		"This is an example query",
		[]string{"this is a document", "this is also a document"},
		voyageai.ModelRerank2Lite,
		nil,
	)
	if err != nil {
		fmt.Printf("Could not get reranking results: %s", err.Error())
	}
	fmt.Printf("Reranking: %v\n", reranking.Data)
	fmt.Printf("Usage: %v", reranking.Usage)
}

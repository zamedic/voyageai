package voyageai

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/gif"
	"image/jpeg"
	"image/png"
	"io"
)

// A list of models supported by the Voyage AI API.
type Model = string

const (
	ModelVoyage3Large      Model = "voyage-3-large"
	ModelVoyage3           Model = "voyage-3"
	ModelVoyage3Lite       Model = "voyage-3-lite"
	ModelVoyage35          Model = "voyage-3.5"
	ModelVoyage35Lite      Model = "voyage-3.5-lite"
	ModelVoyageMultimodal3 Model = "voyage-multimodal-3"
	ModelVoyageCode3       Model = "voyage-code-3"
	ModelVoyageFinance2    Model = "voyage-finance-2"
	ModelVoyageLaw2        Model = "voyage-law-2"
	ModelRerank2           Model = "rerank-2"
	ModelRerank2Lite       Model = "rerank-2-lite"
)

// OutputDimension represents the dimension size for embedding outputs.
type OutputDimension = int

const (
	OutputDimension256  OutputDimension = 256
	OutputDimension512  OutputDimension = 512
	OutputDimension1024 OutputDimension = 1024
	OutputDimension1536 OutputDimension = 1536
	OutputDimension2048 OutputDimension = 2048
)

// A data structure that matches the expected fields of the /embedding endpoint.
// Use [EmbeddingRequestOpts] when building a request for use with [VoyageClient].
// For more details, see the Voyage AI docs "[API reference]."
//
// [API reference]: https://docs.voyageai.com/reference/embeddings-api
type EmbeddingRequest struct {
	// A list of strings to be embedded.
	Input []string `json:"input"`
	// Name of the model. Recommended options: voyage-3-large, voyage-3.5, voyage-3.5-lite, voyage-code-3, voyage-finance-2, voyage-law-2.
	Model string `json:"model"`
	// Type of the input text. Defaults to null. Other options: query, document.
	InputType *string `json:"input_type,omitempty"`
	// Whether to truncate the input texts to fit within the context length. Defaults to true.
	Truncation *bool `json:"truncation,omitempty"`
	// The number of dimensions for resulting output embeddings. Defaults to null.
	OutputDimension *int `json:"output_dimension,omitempty"`
	// The data type for the embeddings to be returned. Defaults to float.
	OutputDType    *string `json:"output_dtype,omitempty"`
	EncodingFormat *string `json:"encoding_format,omitempty"`
}

// Additional request options that can be passed to [VoyageClient.Embed]
type EmbeddingRequestOpts struct {
	InputType       *string `json:"input_type,omitempty"`       // Type of the input text. Defaults to null. Other options: query, document.
	Truncation      *bool   `json:"truncation,omitempty"`       // Whether to truncate the input texts to fit within the context length. Defaults to true.
	OutputDimension *int    `json:"output_dimension,omitempty"` // The number of dimensions for resulting output embeddings. Defaults to null.
	OutputDType     *string `json:"output_dtype,omitempty"`     // The data type for the embeddings to be returned. Defaults to float.
	EncodingFormat  *string `json:"encoding_format,omitempty"`  // Format in which the embeddings are encoded. Defaults to null. Other options: base64.
}

// An embedding object. Part of the data returned by the /embed endpoint
type EmbeddingObject struct {
	Object    string    `json:"object"`    // The object type, which is always "embedding".
	Embedding []float32 `json:"embedding"` // An array of embedding objects.
	Index     int       `json:"index"`     // An integer representing the index of the embedding within the list of embeddings.
}

// Contains details about system usage.
type UsageObject struct {
	TotalTokens int  `json:"total_tokens"`           // The total number of tokens used for computing the embeddings.
	ImagePixels *int `json:"image_pixels,omitempty"` // The total number of image pixels in the list of inputs.
	TextTokens  *int `json:"text_tokens,omitempty"`  // The total number of text tokens in the list of inputs.
}

// The response from the /embed and /multimodalembed endpoints
type EmbeddingResponse struct {
	Object string            `json:"object"` // The object type, which is always "list".
	Data   []EmbeddingObject `json:"data"`   // An array of embedding objects.
	Model  string            `json:"model"`  // Name of the model.
	Usage  UsageObject       `json:"usage"`  // An object containing usage details
}

type text string

// Convert the provided string to the 'text' type for use with [MultimodalInput].
func Text(s string) text {
	return text(s)
}

type imageURL string

// Convert the provided string to the 'imageURL' type for use with [MultimodalInput].
func ImageURL(s string) imageURL {
	return imageURL(s)
}

type imageBase64 string

func imageToBytes(img image.Image, format string) ([]byte, error) {
	buf := new(bytes.Buffer)

	switch format {
	case "png":
		err := png.Encode(buf, img)
		if err != nil {
			return nil, err
		}
	case "jpeg":
		err := jpeg.Encode(buf, img, nil)
		if err != nil {
			return nil, err
		}
	case "gif":
		err := gif.Encode(buf, img, nil)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("cannot encode image of type: %s", format)
	}
	return buf.Bytes(), nil
}

// Reads all image data from an io.Reader and converts it to a base64 encoded data URL for use with [MultimodalInput].
func GetBase64(img io.Reader) (imageBase64, error) {
	dimg, format, err := image.Decode(img)
	if err != nil {
		return "", err
	}

	imgBytes, err := imageToBytes(dimg, format)
	if err != nil {
		return "", err
	}

	imgB64Str := base64.StdEncoding.EncodeToString(imgBytes)

	return imageBase64(fmt.Sprintf("data:image/%s;base64,%s", format, imgB64Str)), nil
}

// Reads all image data and converts it to a base64 encoded data URL for use with [MultimodalInput].
// Panics on failure.
func MustGetBase64(img io.Reader) imageBase64 {
	res, err := GetBase64(img)
	if err != nil {
		panic(err)
	}
	return res
}

// An input for a multimodal embedding request. See [MultimodalEmbed]
type MultimodalInput struct {
	// Specifies the type of the piece of the content. Allowed values are text, image_url, or image_base64.
	Type string `json:"type"`
	// Only present when type is image_url. The value should be a URL linking to the image. We support PNG, JPEG, WEBP, and GIF images.
	Text text `json:"text,omitempty"`
	// Only present when type is image_base64.
	// The value should be a Base64-encoded image in the data URL format data:[<mediatype>];base64,<data>.
	// Currently supported mediatypes are: image/png, image/jpeg, image/webp, and image/gif.
	ImageBase64 imageBase64 `json:"image_base64,omitempty"`
	ImageURL    imageURL    `json:"image_url,omitempty"`
}

// Multimodal returns a new MultimodalInput.
// v must be of type text, imageBase64, or imageURL.
// An empty [MultimodalInput] will be returned for all other types.
func Multimodal(v any) MultimodalInput {
	switch v := v.(type) {
	case text:
		return MultimodalInput{
			Type: "text",
			Text: v,
		}
	case imageBase64:
		return MultimodalInput{
			Type:        "image_base64",
			ImageBase64: v,
		}
	case imageURL:
		return MultimodalInput{
			Type:     "image_url",
			ImageURL: v,
		}
	default:
		return MultimodalInput{}
	}
}

type MultimodalContent struct {
	Content []MultimodalInput `json:"content"`
}

// A data structure that matches the expected fields of the /multimodalembedding endpoint.
// Use [MultimodalRequestOpts] when building a request for use with [VoyageClient].
// For more details, see the Voyage AI docs "[API reference]."
//
// [API reference]: https://docs.voyageai.com/reference/multimodal-embeddings-api
type MultimodalRequest struct {
	Inputs        []MultimodalContent `json:"inputs"`                    // A list of multimodal inputs to be vectorized.
	Model         string              `json:"model"`                     // Name of the model. Currently, the only supported model is voyage-multimodal-3.
	InputType     *string             `json:"input_type,omitempty"`      // Type of the input. Options: None, query, document. Defaults to null.
	Truncation    *bool               `json:"truncation,omitempty"`      // Whether to truncate the inputs to fit within the context length. Defaults to True.
	OuputEncoding *string             `json:"output_encoding,omitempty"` // Format in which the embeddings are encoded. Defaults to null.
}

// Additional request options that can be passed to [VoyageClient.MultimodalEmbed].
type MultimodalRequestOpts struct {
	InputType     *string `json:"input_type,omitempty"`
	Truncation    *bool   `json:"truncation,omitempty"`
	OuputEncoding *string `json:"output_encoding,omitempty"`
}

type APIError struct {
	Detail string `json:"detail"`
}

// A data structure that matches the expected fields of the /rerank endpoint.
// Use [RerankRequestOpts] when building a request for use with [VoyageClient].
// For more details, see the Voyage AI docs "[API reference]."
//
// [API reference]: https://docs.voyageai.com/reference/reranker-api
type RerankRequest struct {
	Query           string   `json:"query"`
	Documents       []string `json:"documents"`
	Model           string   `json:"model"`
	TopK            *int     `json:"top_k,omitempty"`
	ReturnDocuments *bool    `json:"return_documents,omitempty"`
	Truncation      *bool    `json:"truncation,omitempty"`
}

// Additional request options that can be passed to [VoyageClient.Rerank].
type RerankRequestOpts struct {
	// The number of most relevant documents to return. If not specified, the reranking results of all documents will be returned.
	TopK *int `json:"top_k,omitempty"`
	// Whether to return the documents in the response. Defaults to false.
	ReturnDocuments *bool `json:"return_documents,omitempty"`
	// Whether to truncate the input to satisfy the "context length limit" on the query and the documents. Defaults to true.
	Truncation *bool `json:"truncation,omitempty"`
}

// An object containing reranking results.
type RerankObject struct {
	Index          int     `json:"index"`              // The index of the document in the input list.
	RelevanceScore float32 `json:"relevance_score"`    // The relevance score of the document with respect to the query.
	Document       *string `json:"document,omitempty"` // The document string. Only returned when return_documents is set to true.
}

// The response from the /rerank endpoint
type RerankResponse struct {
	Object string         `json:"object"` // The object type, which is always "list".
	Data   []RerankObject `json:"data"`   // An array of the reranking results, sorted by the descending order of relevance scores.
	Model  string         `json:"model"`  // Name of the model.
	Usage  UsageObject    `json:"usage"`  // An object containing usage details
}

---
layout: post
title:  "Sửa lỗi chính tả tự động cho tiếng Việt"
date:   2021-01-20 21:24:58 +0700
# category: projects
image: https://raw.githubusercontent.com/huynhnhathao/myblogs/main/images/vietnamese_spelling_error_correction/hardmasked.png
---

Trong bài viết này, mình sẽ hướng dẫn các bạn xây dựng mô hình Hard-Masked XLM-R cho bài toán sửa lỗi chính tả tự động trên tiếng Việt. Mô hình này sẽ sử dụng một mạng `Bi-LSTM` làm detector và dùng mô hình được huấn luyện sẵn [`XLMRobertaForMaskedLM`][xlm-masklm] làm corrector.

Toàn bộ source code của bài này được viết bằng pytorch và được chia sẻ tại [vietnamese_spelling_error_correction][github_project_link].

## Giới thiệu bài toán

Input của bài toán này sẽ là một câu có độ dài tùy ý có từ sai chính tả hoặc không, bao gồm cả từ viết tắt, teencode,… hoặc có từ không phù hợp trong ngữ cảnh câu nói, đi qua mô hình sửa lỗi sẽ phát hiện ra vị trí của từ sai và đề xuất một từ mới phù hợp với ngữ cảnh dựa vào các từ còn lại. 


*input:  Tôi vẫn luôn **iu** cô ấy với hết tấm lòng của **mk**.*

*output:  Tôi vẫn luôn **yêu** cô ấy với hết tấm lòng của **mình**.*

Với một điều kiện là tỉ lệ từ sai trong câu là khoảng 15%, vì mô hình đề xuất từ vào chỗ sai là mô hình `XLMRobertaForMaskedLM`  đã được huấn luyện sẵn, và mô hình này cần có ngữ cảnh để dự đoán từ thay thế.

Ngoài ra, mô hình này còn có thể dự đoán được lỗi từ đúng chính tả nhưng không phù hợp ngữ cảnh, hoặc vô nghĩa khi đưa vào ngữ cảnh câu. Ví dụ:

*Input: Những ngày cuối tuần đặc biệt của Bitcoin trong thời gian **mèo** đây đang đặt ra thử thách **nước** cho những **cây** chơi tiền điện tử, dù lớn hay nhỏ.*

*Output: Những ngày cuối tuần đặc biệt của Bitcoin trong thời gian **gần** đây đang đặt ra thử thách **mới** cho những **người** chơi tiền điện tử, dù lớn hay nhỏ.*

## Kiến trúc mô hình

| ![HardMaskedXLMR.png](https://raw.githubusercontent.com/huynhnhathao/myblogs/main/images/vietnamese_spelling_error_correction/hardmasked.png) | 
|:--:| 
| *Hard-Masked XLM-R* |


Hình trên mô tả kiến trúc mô hình mà mình sẽ sử dụng cho bài toán này. Kiến trúc này gồm 2 mạng: một mạng detector và một mạng corrector. Một câu input đi qua qua mạng này sẽ trải qua các bước:
1. Input được tokenize thành các token.
2. Các token được chuyển thành các vector embedding, mỗi token có một vector embedding của riêng nó với số chiều bằng nhau và bằng emb_size.
3. Các vector embedding đi qua mạng detector, là một `bidirectional LSTM`. Detector sẽ cho ra xác suất một token là từ sai chính tả của từng token.
4. Các xác suất này được chuyển thành mảng one hot, lấy ngưỡng 0.5, xác xuất lớn hơn 0.5 sẽ thành 1 và nhỏ hơn 0.5 thành 0.
5. Detokenize câu input, sửa lại mảng one hot sao cho phù hợp với câu detokenized.
6. Thay các từ trong câu input ban đầu bằng `<mask>` nếu tại vị trí đó trong mảng one hot bằng 1.
7. Đưa câu mới cho mạng corrector. Mạng corrector sẽ đề xuất từ mới vào `<mask>`.

### Detector
Mạng detector nhận vào một câu và có trách nhiệm chỉ ra vị trí của từ sai trong câu (nếu có). Input của nó là một câu tiếng Việt, output là một mảng one hot có độ dài bằng với số token trong câu (số token của câu sẽ phụ thuộc vào tokenizer được sử dụng), chỉ ra vị trí của từ sai chính tả.

*Input: Con **mòe** có bốn chân*

*Output: 0 1 0 0 0*

Detector là một mạng Bidirectional LSTM,  output tại mỗi time step sẽ đi qua hàm sigmoid để cho ra xác suất bị sai chính tả của token tại time step đó.

| ![bilstm.png](https://raw.githubusercontent.com/huynhnhathao/myblogs/main/images/vietnamese_spelling_error_correction/bilstm.png) | 
|:--:| 
| *Detector là một Bi-LSTM* |

Định nghĩa Detector bằng pytorch:

    class Detector(nn.Module):
        def __init__(self, input_dim,output_dim,  embedding_dim, num_layers, hidden_size):

            super(Detector, self).__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.embedding_dim  = embedding_dim
            self.num_layers = num_layers
            self.hidden_size = hidden_size
            self.embedding = nn.Embedding(num_embeddings = self.input_dim, embedding_dim = self.embedding_dim, )
            self.LSTM = nn.LSTM(input_size = self.embedding_dim, hidden_size= self.hidden_size, num_layers = self.num_layers, 
                                batch_first = True, dropout = 0.1, bidirectional = True)
            self.linear = nn.Linear(self.hidden_size*2, self.output_dim)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            emb = self.embedding(x)
            outputs, (h_n, h_c) = self.LSTM(emb)
            logits = self.linear(outputs)
            p = self.sigmoid(logits)
            return p


Câu input trước khi đi qua detector phải được tokenize trước đó. Trong bài này mình sử dụng sentencepiece tokenizer (mình sẽ giải thích tokenizer này sau). Sau khi tokenize sẽ thu được câu gồm các token,  các token này sẽ đi qua một lớp embedding và trở thành một tensor có số chiều (Tx, emb_dim) với Tx là số token có trong câu, emb_dim là số chiều của mỗi vector embedding. Các vector embedding này đi qua lớp Bi-LSTM và cho ra một chuỗi các hidden_state, là các vector, cho từng time step. Mỗi hidden_state này tiếp tục đi qua một hàm sigmoid sẽ cho ra xác suất của token đó thuộc một từ bị sai.

#### Sentencepiece Tokenizer

[Sentencepiece tokenizer][stp] là một unsupervised tokenizer. Để dùng tokenizer này, nó cần được huấn luyện trên nhiều dữ liệu text.  Ưu điểm của Sentencepiece tokenizer gồm:
1. Nhẹ và nhanh, tránh được trường hợp token unknown do nó chia từ ra thành các subword, nếu nó gặp một từ không nằm trong từ điển thì nó có thể chia từ đó ra đến thành từng chữ cái.
2. Mình có thể xác định trước được số từ vựng cho nó.
3. Việc detokenizer luôn đảm bảo tạo lại câu ban đầu. 

để huấn luyện Sentencepiece tokenizer chỉ cần vài dòng code:  

    !pip install sentencepiece
    import sentencepiece as spm
    spm.SentencePieceTrainer.train('--input="/all_sentences.txt" --model_prefix=spm_tokenizer --vocab_size=10000')

Trong đó dữ liệu input là tập rất nhiều câu tiếng Việt, chưa qua xử lý, mỗi câu trên một dòng. Mình có thể xác định trước
tổng số từ vựng mà tokenizer có qua parameter vocab_size.

### Corrector
Corrector của Hard-Masked XLM-R là mô hình pretrained [`XLMRobertaForMaskedLM`][xlm-masklm], public bởi [huggingface][hgf]. Mô hình này đã được huấn luyện sẵn dựa trên mô hình XLM-RoBERTa. Input của nó là một câu chưa qua xử lý, trong đó có token `<mask>`, output là câu đó với token `<mask>` đã được thay thế bằng một từ khác trong từ điển sao cho phù hợp với ngữ cảnh xung quanh nó. Xét lại ví dụ từ đầu bài:

*Input: Con `<mask>` có bốn chân*

*Output: Con `mèo` có bốn chân*

Tuy nhiên, mô hình `XLMRobertaForMaskedLM` không đảm bảo sẽ cho ra từ chính xác như từ mà mình mong muốn, mà sẽ là bất cứ từ nào nó cho là hợp lý khi thay vào `<mask>`. Như câu trên, `XLMRobertaForMaskedLM` rất có thể sẽ cho ra bất cứ con nào có 4 chân như chó, gấu,... vì không có gì ràng buộc nó để phải output ra con mèo. Để tìm ra từ mong muốn, mình có thể thêm một hàm đo khoảng cách edit distance để tìm ra từ được đề xuất bởi `XLMRobertaForMaskedLM` gần với từ bị sai chính tả nhất, bằng cách này mình sẽ có khả năng cao tìm được từ đúng của từ sai chính tả ban đầu.

Nói một chút về [XLM-RoBERTa][xlmr], do Facebook phát triển, là một mô hình ngôn ngữ đã được huấn luyện trên 100 ngôn ngữ, trong đó có tiếng Việt. Mình có thể xem mô hình ngôn ngữ này như một mô hình embedding, vì nó chuyển các token thành các vector số để đại diện cho token đó. Trong xử lý ngôn ngữ tự nhiên, việc tìm một vector đại diện cho một từ trong ngôn ngữ để máy có thể hiểu và xử lý được là rất quan trọng. Nếu vector đại diện đó tốt, thể hiện được ý nghĩa của từ mà nó đại diện cho thì mô hình học máy phía sau sẽ cho kết quả tốt hơn. Các mô hình ngôn ngữ có hỗ trợ tiếng Việt phải kể đển PhoBERT, XLM-RoBERTa, BERT-multilingual. Nhưng trong đó XLM-R được huấn luyện trên nhiều dữ liệu tiếng Việt nhất (khoảng 137GB tiếng Việt), nên XLM-R rất có giá trị cho các bài toán xử lý ngôn ngữ tự nhiên cho tiếng Việt.

### Huấn luyện mô hình

Mình chỉ cần huấn luyện detector, vì corrector đã được huấn luyện sẵn. Detector được huấn luyện trên 2 triệu câu tiếng Việt, qua 2 epochs, và đạt được f1-score 96.7% trên tập dev. 

Vì tiếng Việt chưa có dữ liệu được gán nhãn và public cho bài sửa lỗi chính tả nên toàn bộ dữ liệu training và dev đều được tạo từ một hàm synthesized function. Dữ liệu được tạo bởi hàm này có thể không thể hiện tốt được phân phối của dữ liệu trong thực tế, nhưng mình đã cố gắng tạo hàm synthesized function sao cho nó tạo ra những lỗi mà người viết hay gặp nhất, như lỗi từ đồng âm, lỗi đánh máy, từ viết tắt, teencode,...

### Kết luận

Trong bài này, mình đã mô tả mô hình Hard-Masked XLM-R để giải bài toán sửa lỗi chính tả cho tiếng Việt. Mô hình này dùng một detector là một mạng `Bi-LSTM` để tìm ra vị trí của từ sai, và đề xuất từ sửa lỗi bằng mô hình huấn luyện sẵn `XLMRobertaForMaskedLM`. Dữ liệu được sử dụng là dữ liệu tổng hợp từ một hàm synthesized function. Sau khi huấn luyện trên 2 triệu câu tiếng Việt, detector đạt được f1-score hơn 96%. Toàn bộ source code để chạy lại bài này đã được mình public tại [github][github_project_link].

[xlm-masklm]: https://huggingface.co/transformers/model_doc/xlmroberta.html#xlmrobertaformaskedlm
[github_project_link]: https://github.com/huynhnhathao/vietnamese_spelling_error_correction
[stp]: https://github.com/google/sentencepiece
[xlmr]: https://arxiv.org/abs/1911.02116
[hgf]: https://huggingface.co/
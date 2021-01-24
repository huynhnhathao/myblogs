---
layout: post
title:  "Sửa lỗi chính tả tự động cho tiếng Việt"
date:   2021-01-20 21:24:58 +0700
# category: projects
image: https://raw.githubusercontent.com/huynhnhathao/myblogs/main/images/vietnamese_spelling_error_correction/hardmasked.png
---

Trong bài viết này, mình sẽ hướng dẫn các bạn xây dựng mô hình detector-corrector cho bài toán sửa lỗi chính tả tự động trên tiếng Việt. Mô hình này sẽ sử dụng một mạng Bi-LSTM làm detector và dùng mô hình được huấn luyện sẵn [`XLMRobertaForMaskedLM`][xlm-masklm] làm corrector.

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


Hình trên mô tả kiến trúc mô hình mà mình sẽ sử dụng cho bài toán này. Kiến trúc này gồm 2 mạng: một mạng detector và một mạng corrector.

### Detector
Mạng detector nhận vào một câu và có trách nhiệm chỉ ra vị trí của từ sai trong câu (nếu có). Input của nó là một câu tiếng Việt, output là một mảng one hot có độ dài bằng với số token trong câu (số token của câu sẽ phụ thuộc vào tokenizer được sử dụng), chỉ ra vị trí của từ sai chính tả.

*Input: Con mòe có bốn chân.*

*Output: 0 1 0 0 0*

Detector là một mạng Bidirectional LSTM,  output tại mỗi time step sẽ đi qua hàm sigmoid để cho ra xác suất bị sai chính tả của token tại time step đó.

| ![bilstm.png](https://raw.githubusercontent.com/huynhnhathao/myblogs/main/images/vietnamese_spelling_error_correction/hardmasked.png) | 
|:--:| 
| *Detector là một Bi-LSTM* |

Detector được huấn luyện trên rất nhiều câu tiếng Việt có từ sai và nhãn one hot của nó. 











[xlm-masklm]: https://huggingface.co/transformers/model_doc/xlmroberta.html#xlmrobertaformaskedlm
[github_project_link]: https://github.com/huynhnhathao/vietnamese_spelling_error_correction
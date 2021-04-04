---
layout: post
title:  "Tài liệu học Machine Learning"
date:   2021-04-04 2:00:00 +0700
# category: projects
image: https://raw.githubusercontent.com/huynhnhathao/myblogs/main/images/learn_machine_learning/time_treasure.png
---
Chào các bạn, trong bài viết này mình sẽ giới thiệu về các tài liệu quý giá để học machine learning, deep learning mà mình đã tìm hiểu, chắt lọc. Con đường học machine learning là không dễ, cần rất nhiều cố gắng và nỗ lực. Tuy nhiên, những ai có đam mê, khao khát tri thức sẽ cảm thấy đây là một hành trình thú vị. Machine learning gồm rất nhiều lĩnh vực liên quan, như xác xuất, đại số tuyến tính, tối ưu hóa, thống kê,… mình sẽ chia sẽ tất cả các sách, khóa học, trang web, … mà mình biết.

## 1. Kiến thức nền tảng

Toán luôn làm người học thấy chán nản không những vì nó khó, mà nó còn không cho ta thấy được ứng dụng ngay lập tức. Tuy nhiên, Albert Enstein đã nói, `Toán học là ông vua của mọi ngành khoa học`, toán là cái nền của machine learning, của mọi thuật toán trong lĩnh vực. Khi ta tìm hiểu các thuật toán trong deep learning như CNN, RNN, ANN,… tất cả các thuật toán đều có một điểm chung là tìm các tham số cho mô hình bằng việc tối ưu hàm loss, thông qua đạo hàm của nó. Các môn toán quan trọng cần biết gồm đại số tuyến tính, xác xuất thống kê.

### 1. Đại số tuyến tính

* Khóa học linear algebra của Gilbert Strang, MIT trên [youtube][GS_LinearAlgebra]

* Sách [Introduction to Linear Algebra][GS_book], Fifth Edition (2016) , cũng của Gilbert Strang

* Hoặc khóa học MIT 18.065 [Matrix Methods in Data Analysis, Signal Processing, and Machine Learning, Spring 2018, Gilbert Strang][GS_LinearAlgebra2]

### 2. Xác xuất thống kê

* Khóa học Statistics 110: [Probability, Joe Blitzstein, Havard University][Probability_course]

* Cùng với quyển sách đồng hành rất hay với khóa học: [Introduction to Probability by Joe Blitzstein and Jessica Hwang][Probability_book]

## 2. Machine Learning

### 1. Machine learning cơ bản

#### Tài liệu Tiếng Việt

Quyển đầu tiên về machine learning nhất định phải đọc là `machine learning cơ bản` của anh Vũ Hữu Tiệp, cùng với trang web [Machine learning cơ bản][ml_coban] có rất nhiều thông tin quý giá về machine learning. Bạn đọc không nên bỏ qua nguồn tài liệu quý giá này.

#### Tài liệu Tiếng Anh

*Machine learning*

Các bạn nào có thể đọc tiếng anh thì mình giới thiệu 2 quyển:

* An Introduction to Statistical Learning: With Applications in R (ISRL), Daniela Witten, Trevor Hastie, Gareth M. James, Robert Tibshirani

* The Elements of Statistical Learning(ESL), Book by Jerome H. Friedman, Robert Tibshirani, and Trevor Hastie

Quyển ISLR giới thiệu về các thuật toán machine learning cũng như quyển ESL, tuy nhiên ISLR giới thiệu một cách nhẹ nhàng, ít các phương trình toán học hơn, nên sẽ dễ đọc hơn. Mình khuyên các bạn nào mới học nên đọc quyển ISLR trước, sau đó nếu muốn tìm hiểu sâu hơn thì đọc quyển ESL. Cả 2 quyển sách đều rất hay.

Sau khi biết được các khái niệm cơ bản về thống kê và các thuật toán trong 2 quyển trên, mình xin giới thiệu quyển 
* Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems, Book by Aurelien Geron, second edition. 

Quyển này có rất nhiều code bằng python, thiên về hướng `practical` hơn 2 quyển trước. Các bạn có thể tìm hiểu cách code trong quyển này và bắt chước theo. Đây là một quyển rất hay, không thể bỏ qua.

* Ngoài ra còn có khóa học machine learning trên Coursera, do Prof. Andrew Ng., đại học Standford dạy. Các bạn có thể học thoải mái không tốn tiền trên Coursera nên đừng ngại lên đó tìm khóa học mà mình muốn. [Machine Learning, Coursera][ml_coursera]

Nếu các bạn muốn có giấy chứng nhận sau khi hoàn thành khóa học mà không có tiền để trả phí, thì có thể đăng ký Financial Aid của Coursera. Mình đảm bảo sẽ được chấp nhận nếu bạn tự viết application, không copy trên mạng, và một khi đã được chấp nhận thì phải hoàn thành khóa học đã đăng ký.

* Hoặc các bạn có thể học khóa machine learning CS229 trên youtube, cũng của Andrew Ng., Chất lượng Stanford [Stanford CS229: Machine Learning Autumn 2018 - YouTube][ml_youtube]

*Deep learning*

Đối với deep learning, các khóa đầu tiên mình giới thiệu các bạn nên học là 5 khóa deep learning của Coursera [Deep Learning - Coursera][dl_coursera]

Nhắc lại là các bạn có thể học miễn phí các khóa này.
Tiếp theo là các khóa học trên youtube của Stanford University

* CS230: [Deep Learning  Autumn 2018 - YouTube][dl_youtube], rất tuyệt vời

* CS224N: [Natural Language Processing with Deep Learning - YouTube][nlp_youtube], dành cho bạn nào muốn học NLP, tuyệt vời không kém

* [Lecture Collection  Convolutional Neural Networks for Visual Recognition (Spring 2017) - YouTube][cv_youtube], cho bạn nào muốn học CV

Ngoài ra, một số quyển sách nói sâu về Deep learning:

* [Deep Learning, Book by Aaron Courville, Ian Goodfellow, and Yoshua Bengio][dl_book]. Quyển này sẽ hơi khó đọc với các bạn mới học.

* [Deep Learning with Python, Book by François Chollet][dl_keras]. Quyển này dễ đọc, nói về keras, một thư viện deep learning thân thiện với người mới.

* Một quyển rất hay và chi tiết về NLP: [Speech and language processing][slp_book]

Các thư viện deep learning như Pytorch, Keras, Tensorflow có nhiều bài viết tutorials trên trang chính, các bạn có thể lên đó tham khảo.
Reinforcement learning

Bạn nào muốn tìm hiểu về reinforcement learning có thể tham khảo các nguồn tài liệu sau

* Reinforcement Learning: An Introduction, Book by Andrew Barto and Richard S. Sutton

* Reinforcement Learning Specialization, [Reinforcement Learning - Coursera][rl_coursera]

* Foundations of Deep Reinforcement Learning: Theory and Practice in Python, Book by Laura Graesser and Wah Loon Keng

* Khóa học RL trên youtube: [Introduction to reinforcement learning - YouTube, David Silver][rl_silver]

* Khóa học DRL trên youtube: [CS294-112 Deep Reinforcement Learning Sp17 - YouTube][drl_youtube]




## Kết luận

Các tài liệu mà mình giới thiệu đều là Tiếng Anh. Cho nên, Tiếng Anh rất quan trọng đối với lĩnh vực này, nó là phương tiện giúp các bạn mở rộng tri thức. Mình khuyên các bạn nào chưa đọc, nghe tốt Tiếng Anh hãy bắt đầu học ngay lập tức bằng cách đọc sách Tiếng Anh, nghe Tiếng Anh và đừng lấy bất cứ lý do nào để trì hoãn, vì một hành trình 1000 bậc thang cũng phải bắt đầu ở bậc thang số 1. Một cách học Tiếng Anh rất có hiệu quả đối với mình là đọc thẳng vào một quyển sách Tiếng Anh hàng trăm trang, hay học một khóa học đã giới thiệu ở trên, dừng lại ở những từ mà mình không biết, dùng Google dịch để tra, và tiếp tục xem tiếp. Lúc mới bắt đầu, một video 10 phút nói Tiếng Anh có thể lấy của mình cả ngày để hiểu. Tất cả những gì các bạn cần là đam mê và kiên trì, với hai thứ này bạn có thể học được bất cứ thứ gì mà bạn muốn.

Điều cuối cùng mình muốn chia sẻ là đừng lãng phí thời gian của mình, nhất là thời gian học đại học. Hãy dùng nó vào những gì có ích.

Mình sẽ cập nhật các tài liệu mà mình biết vào bài viết này, nếu có thắc mắc hãy liên lạc với mình qua Gmail, Linkedin.




[GS_LinearAlgebra]: https://www.youtube.com/playlist?list=PL221E2BBF13BECF6C
[GS_book]: https://math.mit.edu/~gs/linearalgebra/
[GS_LinearAlgebra2]: https://www.youtube.com/playlist?app=desktop&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k
[Probability_course]: https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo
[Probability_book]: https://projects.iq.harvard.edu/stat110/home
[ml_coban]: https://machinelearningcoban.com/
[ml_coursera]: https://www.coursera.org/learn/machine-learning?utm_source=gg&utm_medium=sem&utm_campaign=07-StanfordML-ROW&utm_content=07-StanfordML-ROW&campaignid=2070742271&adgroupid=80109820241&device=c&keyword=machine%20learning%20mooc&matchtype=b&network=g&devicemodel=&adpostion=&creativeid=369041663186&hide_mobile_promo&gclid=CjwKCAjwpKCDBhBPEiwAFgBzj1iCJwBA1rcJCawcZUpGrpzUHn7Ol53V-rzLP9anDsk43w8EL8cTxxoCsMQQAvD_BwE
[ml_youtube]: https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU
[dl_coursera]: https://www.coursera.org/specializations/deep-learning
[dl_youtube]: https://www.youtube.com/playlist?list=PLoROMvodv4rOABXSygHTsbvUz4G_YQhOb
[nlp_youtube]: https://www.youtube.com/playlist?list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z
[cv_youtube]: https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv
[dl_book]: https://www.deeplearningbook.org/
[dl_keras]: http://faculty.neu.edu.cn/yury/AAI/Textbook/Deep%20Learning%20with%20Python.pdf
[rl_coursera]: https://www.coursera.org/specializations/reinforcement-learning
[rl_silver]: https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
[drl_youtube]: https://www.youtube.com/playlist?list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX
[slp_book]: https://web.stanford.edu/~jurafsky/slp3/
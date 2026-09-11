# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** Chưa xác nhận CPU/GPU

**Python / PyTorch / Ultralytics:** Python: chưa cung cấp; PyTorch: chưa cung cấp; Ultralytics: `8.4.145`

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Chưa xác nhận có thay đổi mã nguồn. Tuy nhiên, cần kiểm tra lại dữ liệu đầu ra vì `detection_predictions.json` và `segmentation_predictions.json` được cung cấp có `sample_id = "traffic"`, trong khi yêu cầu báo cáo chỉ định sample `kitchen` cho hai tác vụ này.

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
>
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
>
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
>
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): `class_id = 468`; `class_name = "cab"`; `rank = 1`; `score = 0.510915`; `taxonomy_name = "ImageNet-1K"`.

- Record này mô tả toàn ảnh như thế nào? Mô hình phân loại toàn ảnh `traffic` là `cab` với score `0.510915`, tức khoảng 51,09%. Kết quả cho thấy đặc trưng xe taxi hoặc xe chở khách là đặc trưng nổi bật nhất mà mô hình nhận ra trong toàn cảnh giao thông. Đây là prediction cấp ảnh, không phải nhãn riêng cho từng phương tiện hay người xuất hiện trong ảnh.

- Ai định nghĩa class list mà checkpoint có thể dự đoán? Class list của checkpoint `yolo11n-cls.pt` được xác định bởi taxonomy của bộ dữ liệu dùng để huấn luyện mô hình; trong kết quả này là `ImageNet-1K`. Ánh xạ giữa `class_id` và `class_name` được lưu kèm checkpoint/metadata của mô hình. Người đọc kết quả phải sử dụng đúng danh sách đó, không tự đổi ID hoặc đặt tên lớp mới.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? `class_id` giúp máy xử lý và đối chiếu ổn định; `class_name` giúp con người hiểu nhãn; `taxonomy_name` xác định hệ thống lớp mà ID và tên lớp đang thuộc về. Giữ đủ ba trường giúp tránh nhầm cùng một ID giữa các bộ dữ liệu khác nhau và bảo đảm khả năng truy vết kết quả.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline cần quy định tác vụ là single-label hay multi-label, tiêu chí chọn chủ thể chính, cách xử lý các chủ thể có mức nổi bật tương đương và trường hợp nào phải gắn nhãn mơ hồ hoặc chuyển reviewer quyết định.

- Vì sao model score không phải ground truth? Score `0.510915` chỉ thể hiện mức tự tin của mô hình đối với lớp `cab`, không chứng minh đó là nhãn đúng tuyệt đối. Kết quả top 5 còn có `minibus`, `police_van`, `recreational_vehicle` và `streetcar`, cho thấy mô hình vẫn có độ bất định. Ground truth phải được annotator tạo hoặc con người xác minh theo guideline đã thống nhất.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`): `class_name = "bus"`; `score = 0.912558`; `bbox_xyxy = [93.17, 187.95, 223.01, 320.91]`; `bbox_width = 129.84`; `bbox_height = 132.96`. Record được cung cấp hiện có `sample_id = "traffic"`, chưa khớp với sample `kitchen` nêu trong yêu cầu nên cần kiểm tra lại trước khi nộp.

- Diễn giải vị trí box bằng lời: Box bao quanh một xe buýt nằm ở nửa bên trái và khu vực giữa–dưới của ảnh 640 × 428 pixel. Góc trên trái của box ở tọa độ `(93.17, 187.95)` và góc dưới phải ở `(223.01, 320.91)`. Box rộng 129.84 pixel, cao 132.96 pixel, tương đương khoảng 20,3% chiều rộng và 31,1% chiều cao của ảnh.

- So sánh số prediction ở hai threshold: Ở threshold `0.35`, checkpoint `yolo11n.pt` phát hiện `53` prediction. Kết quả ở threshold thứ hai chưa được cung cấp, vì vậy chưa thể ghi số lượng hoặc mức chênh lệch chính xác. Không dùng số `49` của `yolo11n-seg.pt` để thay thế vì đó là số instance của một tác vụ và checkpoint khác, không phải detection tại threshold thứ hai.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Khi hạ threshold, mô hình giữ lại nhiều prediction hơn, giúp tăng độ bao phủ và khả năng phát hiện vật thể nhỏ, xa hoặc bị che khuất; đồng thời false positive và khối lượng reviewer phải kiểm tra có thể tăng. Khi tăng threshold, số prediction và khối lượng kiểm tra thường giảm, nhưng nguy cơ bỏ sót vật thể cũng tăng.

- Đề xuất một quy tắc box chặt: Mỗi box chỉ bao một object instance, bám sát phần nhìn thấy của vật thể, không cắt vào vật thể và không chứa nền thừa đáng kể. Tọa độ box phải nằm trong giới hạn ảnh; không gộp nhiều vật thể vào một box và không tạo box trùng cho cùng một instance.

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? Guideline phải quy định tỷ lệ nhìn thấy tối thiểu để annotate, box chỉ bao phần nhìn thấy hay ước lượng phần bị che, cách xử lý vật thể bị cắt tại mép ảnh và ngưỡng kích thước tối thiểu. Nếu không xác định chắc lớp hoặc ranh giới, annotator phải chuyển reviewer hoặc người phụ trách guideline quyết định thay vì tự suy đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`): `instance_id = "traffic-001"`; `class_name = "bus"`; `score = 0.925745`; số điểm polygon = `120`; một phần `polygon_xy = [[148.0, 189.0], [147.0, 190.0], [145.0, 190.0], [143.0, 192.0], [142.0, 192.0], ...]`. Ở threshold `0.35`, file được cung cấp ghi nhận tổng cộng `49` instance. Record JSON này thuộc sample `traffic`, trong khi hình segmentation được cung cấp là cảnh `kitchen`; hai evidence chưa cùng một sample nên không được xem là một cặp JSON–visual hoàn toàn khớp.

- Polygon bổ sung chi tiết gì so với box? Box chỉ mô tả một hình chữ nhật bao quanh vật thể, còn polygon bám theo đường biên của vật thể nên thể hiện được hình dạng, diện tích và phần nền không thuộc vật thể chính xác hơn.

- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` dùng để phân biệt các cá thể riêng biệt trong cùng một ảnh, kể cả khi chúng thuộc cùng một class. Đây không phải `class_id`, không phải ID taxonomy và cũng không phải mã định danh lâu dài của một vật thể ngoài đời hoặc xuyên qua nhiều ảnh.

- Đề xuất một quy tắc biên mask: Mask phải bám theo đường biên nhìn thấy của từng instance ở mức pixel hợp lý, không lấn sang nền hoặc instance kế bên, không bỏ sót phần nhìn thấy rõ và không tự suy diễn phần bị che khuất nếu guideline không yêu cầu.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? Guideline cần quy định mức độ rõ tối thiểu của biên, cách tách hai instance đang tiếp xúc, cách xử lý lỗ rỗng, bóng đổ, vật thể trong suốt và phần bị che khuất. Trường hợp biên không thể xác định nhất quán hoặc class không rõ phải chuyển reviewer quyết định.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn hoặc tập nhãn cấp ảnh, gồm class ID/tên lớp theo taxonomy được quy định | Ảnh `traffic` có nhiều xe và nhiều loại chủ thể, nhưng mô hình phải chọn một lớp cấp ảnh; top 1 là `cab` với score chỉ khoảng 51,09% nên vẫn có sự mơ hồ với `minibus`, `police_van`, `recreational_vehicle` và `streetcar` | Đọc guideline, xem toàn ảnh, chọn đúng lớp theo tiêu chí chủ thể chính và đánh dấu trường hợp nhiều chủ thể hoặc không chắc chắn để escalation | Kiểm tra nhãn có đúng chủ thể chính, đúng taxonomy `ImageNet-1K`, nhất quán với guideline và không biến model score thành ground truth |
| Phát hiện vật thể | Một class label và một bounding box cho mỗi object instance, theo định dạng `xyxy`, đơn vị pixel | Ở threshold `0.35` có 53 prediction nên reviewer phải kiểm tra nhiều box; có object nhỏ/cắt mép như `person` với box bắt đầu tại `x = 0.08`; ngoài ra JSON mang sample `traffic` nhưng yêu cầu ghi sample `kitchen` | Tạo một box chặt cho mỗi instance đủ điều kiện, gắn đúng class, kiểm tra vật thể nhỏ/cắt mép và chuyển trường hợp không chắc chắn | Kiểm tra độ bao phủ, độ chặt của box, đúng lớp, box trùng, object bị bỏ sót, cách xử lý che khuất/cắt mép và tính khớp giữa sample trong JSON với visual |
| Instance segmentation | Một class label và một polygon/mask riêng cho mỗi instance | Hình `kitchen` có nhiều vùng tiếp xúc hoặc che nhau như các bowl trên bàn, bàn với các vật đặt phía trên và người đứng trước nền phức tạp; một số prediction có score thấp. Quan trọng hơn, hình là `kitchen` nhưng JSON được cung cấp là `traffic` | Vẽ mask bám theo phần nhìn thấy, tách riêng từng instance, không lấn sang nền hoặc vật thể kế bên và báo escalation khi biên không rõ | Kiểm tra đường biên khi phóng to, phần lấn/thiếu, tách instance, polygon hợp lệ và xác nhận JSON–visual thuộc cùng sample trước khi chấp nhận |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ sử dụng ảnh và dữ liệu được cấp phép cho bài thực hành; không đưa dữ liệu cá nhân hoặc dữ liệu nhạy cảm vào notebook, output, báo cáo hay repository công khai.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: Giảng viên hoặc người phụ trách học phần/dataset để được xác minh trước khi tiếp tục xử lý.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json` — nội dung prediction top 5 đã được cung cấp; cần bảo đảm file thật nằm đúng thư mục khi nộp.

- [x] `detection_predictions.json` — đã có thông tin threshold `0.35`, tổng 53 prediction và các record mẫu; cần kiểm tra lại sample `traffic/kitchen`.

- [x] `segmentation_predictions.json` — đã có thông tin tổng 49 instance và record polygon; cần kiểm tra lại sample `traffic/kitchen`.

- [x ] `IMAGE_ATTRIBUTION.md`

- [ x] `visuals/classification_top5.png`

- [ x] `visuals/detection_predictions.png`

- [ x] `visuals/segmentation_prediction.png` — đã có hình minh họa cảnh `kitchen`, nhưng file được cung cấp tên `anh day1.png`; cần đổi/đặt đúng tên và xác nhận khớp JSON.

- [ x] Ô validation cuối notebook báo `PASS`.

- [ x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.

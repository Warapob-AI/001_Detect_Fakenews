from pythainlp.corpus import thai_stopwords, thai_words
from pythainlp.tokenize import Tokenizer
from pythainlp.util import dict_trie
import re

class NeuroLinguisticProgramming: 
    def load_data(self, filename): 
        import codecs

        with codecs.open(filename, 'r', 'utf-8') as file: 
            return [line.strip() for line in file.readlines()]

    def Lemmatization(self, data_list): 
        word_lemmatization = {
            r'กก\.บห\.?': 'คณะกรรมการบริหาร',
            r'กกบห\.': 'คณะกรรมการบริหาร',

            r'ส\.ป\.ส\.ช\.?': 'สำนักงานหลักประกันสุขภาพแห่งชาติ',
            r'สปสช\.': 'สำนักงานหลักประกันสุขภาพแห่งชาติ',

            r'ก\.ส\.ท\.ช\.?': 'คณะกรรมการกิจการกระจายเสียง กิจการโทรทัศน์ และกิจการโทรคมนาคมแห่งชาติ',
            r'กสทช\.': 'คณะกรรมการกิจการกระจายเสียง กิจการโทรทัศน์ และกิจการโทรคมนาคมแห่งชาติ',

            r'ร\.ม\.ว\.?': 'รัฐมนตรีว่าการ',
            r'รมว\.': 'รัฐมนตรีว่าการ',

            r'ส\.ค\.ร\.?': 'สำนักงานคณะกรรมการนโยบายรัฐวิสาหกิจ',
            r'สคร\.': 'สำนักงานคณะกรรมการนโยบายรัฐวิสาหกิจ',

            r'ค\.ร\.ม\.?': 'คณะรัฐมนตรี',
            r'ครม\.': 'คณะรัฐมนตรี',

            r'ส\.ม\.ช\.?': 'สำนักงานสภาความมั่นคงแห่งชาติ',
            r'สมช\.': 'สำนักงานสภาความมั่นคงแห่งชาติ',

            r'น\.ร\.ต\.?': 'นักเรียนนายร้อยตำรวจ',
            r'นรต\.': 'นักเรียนนายร้อยตำรวจ',

            r'ส\.ส\.ส\.?': 'สำนักงานกองทุนสนับสนุนการสร้างเสริมสุขภาพ',
            r'สสส\.': 'สำนักงานกองทุนสนับสนุนการสร้างเสริมสุขภาพ',

            r'ผ\.บ\.ช\.': 'ผู้บัญชาการ',
            r'ผบช\.': 'ผู้บัญชาการ',

            r'ต\.ร\.?': 'ตำรวจ',
            r'ตร\.': 'ตำรวจ',

            r'ส\.ธ\.?': 'กระทรวงสาธารณสุข',
            r'สธ\.': 'กระทรวงสาธารณสุข',

            r'ก\.อ\.ช\.?': 'กองทุนการออมแห่งชาติ',
            r'กอช\.': 'กองทุนการออมแห่งชาติ',

            r'ส\.ป\.ป\.?': 'สาธารณรัฐประชาธิปไตยประชาชน',
            r'สปป\.': 'สาธารณรัฐประชาธิปไตยประชาชน',

            r'ก\.ส\.ร\.?': 'กรมสวัสดิการและคุ้มครองแรงงาน',
            r'กสร\.': 'กรมสวัสดิการและคุ้มครองแรงงาน',

            r'ก\.ท\.ม\.?': 'กรุงเทพมหานคร',
            r'กทม\.': 'กรุงเทพมหานคร',

            r'ก\.ก\.ต\.?': 'คณะกรรมการการเลือกตั้ง',
            r'กกต\.': 'คณะกรรมการการเลือกตั้ง',

            r'อ\.บ\.จ\.?': 'องค์การบริหารส่วนจังหวัด',
            r'อบจ\.': 'องค์การบริหารส่วนจังหวัด',

            r'พ\.ร\.บ\.?': 'พระราชบัญญัติ',
            r'พรบ\.': 'พระราชบัญญัติ',

            r'ผ\.ก\.ก\.?': 'ผู้กำกับการ',
            r'ผกก\.': 'ผู้กำกับการ',

            r'ก\.ก\.ร\.?': 'คณะกรรมการร่วมภาคเอกชน 3 สถาบัน',
            r'กกร\.': 'คณะกรรมการร่วมภาคเอกชน 3 สถาบัน',

            r'ส\.พ\.ฐ\.?': 'สำนักงานคณะกรรมการการศึกษาขั้นพื้นฐาน',
            r'สพฐ\.': 'สำนักงานคณะกรรมการการศึกษาขั้นพื้นฐาน',

            r'ม\.ท\.ร\.?': 'มหาวิทยาลัยเทคโนโลยีราชมงคล',
            r'มทร\.': 'มหาวิทยาลัยเทคโนโลยีราชมงคล',

            r'จ\.น\.ท\.?': 'เจ้าหน้าที่',
            r'จนท\.': 'เจ้าหน้าที่',

            r'ป\.ป\.ง\.?': 'สำนักงานป้องกันและปราบปรามการฟอกเงิน',
            r'ปปง\.': 'สำนักงานป้องกันและปราบปรามการฟอกเงิน',

            r'ธ\.ก\.ส\.?': 'ธนาคารเพื่อการเกษตรและสหกรณ์การเกษตร',
            r'ธกส\.': 'ธนาคารเพื่อการเกษตรและสหกรณ์การเกษตร',

            r'ส\.ป\.ส\.?': 'ธนาคารเพื่อการเกษตรและสหกรณ์การเกษตร',
            r'ธกส\.': 'ธนาคารเพื่อการเกษตรและสหกรณ์การเกษตร',

            r'พ\.น\.ง\.?': 'พนักงาน',
            r'พนง\.': 'พนักงาน',

            r'ก\.ส\.ท\.?': 'การสื่อสารแห่งประเทศไทย',
            r'กสท\.': 'การสื่อสารแห่งประเทศไทย',

            r'ก\.ฟ\.ภ\.?': 'การไฟฟ้าส่วนภูมิภาค',
            r'กฟภ\.': 'การไฟฟ้าส่วนภูมิภาค',

            r'ต\.ก\.?': 'กองแผนงานกิจการพิเศษ',
            r'ตก\.': 'กองแผนงานกิจการพิเศษ',

            r'พ\.ม\.?': 'กระทรวงการพัฒนาสังคมและความมั่นคงของมนุษย์',
            r'พม\.': 'กระทรวงการพัฒนาสังคมและความมั่นคงของมนุษย์',
            
            r'ก\.ม\.?': 'กฎหมาย',
            r'กม\.': 'กฎหมาย',

            r'ม\.ท\.?': 'กฎหมาย',
            r'มท\.': 'กฎหมาย',

            r'อ\.ย\.?': 'กระทรวงมหาดไทย',
            r'อย\.': 'กระทรวงมหาดไทย',

            r'ผ\.อ\.?': 'ผู้อำนวยการ',
            r'ผอ\.': 'ผู้อำนวยการ',

            r'ผ\.ก\.?': 'กองแผนงานกิจการพิเศษ',
            r'ผก\.': 'กองแผนงานกิจการพิเศษ',

            r'ร\.พ\.?': 'โรงพยาบาล',
            r'รพ\.': 'โรงพยาบาล',

            r'ส\.ภ\.?': 'สถานีตำรวจภูธร',
            r'สภ\.': 'สถานีตำรวจภูธร',

            r'น\.ร\.?': 'นักเรียน',
            r'นร\.': 'นักเรียน',
            
            r'ร\.ร\.?': 'โรงเรียน',
            r'รร\.': 'โรงเรียน',

            r'ท\.ส\.?': 'กระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม',
            r'ทส\.': 'กระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม',

            r'น\.ช\.?': 'นักโทษชาย',
            r'นช\.': 'นักโทษชาย',

            r'ศ\.ธ\.?': 'กระทรวงศึกษาธิการ',
            r'ศธ\.': 'กระทรวงศึกษาธิการ',

            r'ผ\.บ\.?': 'ผู้บัญชาการ',
            r'ผบ\.': 'ผู้บัญชาการ',

            r'ม\.ค\.?': 'มกราคม', 
            r'มค\.': 'มกราคม',

            r'ก\.พ\.?': 'กุมภาพันธ์',
            r'กพ\.': 'กุมภาพันธ์',

            r'มี\.ค\.?': 'มีนาคม',
            r'มีค\.': 'มีนาคม',

            r'เม\.ย\.?': 'เมษายน',
            r'เมย\.': 'เมษายน',

            r'พ\.ค\.?': 'พฤษภาคม',
            r'พค\.': 'พฤษภาคม',

            r'มิ\.ย\.?': 'มิถุนายน',
            r'มิย\.': 'มิถุนายน',

            r'ก\.ค\.?': 'กรกฎาคม',
            r'กค\.': 'กรกฎาคม',

            r'ส\.ค\.?': 'สิงหาคม',
            r'สค\.': 'สิงหาคม',

            r'ก\.ย\.?': 'กันยายน',
            r'กย\.': 'กันยายน',

            r'ต\.ค\.?': 'ตุลาคม',
            r'ตค\.': 'ตุลาคม',

            r'พ\.ย\.?': 'พฤศจิกายน',
            r'พย\.': 'พฤศจิกายน',

            r'ธ\.ค\.?': 'ธันวาคม',
            r'ธค\.': 'ธันวาคม',

            r'ม\.': 'มาตรา',

            r'จ\.': 'จังหวัด',
        }

        for pattern, full_month in word_lemmatization.items():
            data_list = [re.sub(pattern, full_month, t) for t in data_list]

        return data_list
    
    
    
    def Remove_Number(self, data_list): 
        text_no_numbers = [re.sub(r'\d+', '', t) for t in data_list]
        return text_no_numbers
    
    def Remove_Special_character(self, data_list): 
        cleaned_data_list = [re.sub(r'[^a-zA-Z0-9ก-ฮะ-์]+', '', t) for t in data_list]
        return cleaned_data_list
    
    def Tokenization(self, data_list): 
        custom_words_list = set(thai_words())
        custom_words_list.add('รัฐมนตรีว่าการ')
        custom_words_list.add('สำนักงานกองทุนสนับสนุนการสร้างเสริมสุขภาพ')
        custom_words_list.add('ตำรวจไซเบอร์')
        custom_words_list.add('ไม้มิสวาก')
        custom_words_list.add('ไม่มี')
        custom_words_list.add('คอล')
        custom_words_list.add('ผู้บัญชาการไซเบอร์')
        custom_words_list.add('ของฟรีไม่มีในโลก')
        custom_words_list.add('ล้านบาท')
        custom_words_list.add('ทุบสถิติ')
        custom_words_list.add('ไม่ให้')
        custom_words_list.add('แสนราย')
        custom_words_list.add('พันล้าน')
        custom_words_list.add('กาซ่า')
        custom_words_list.add('กองทุนการออมแห่งชาติ')
        custom_words_list.add('ต้องการ')
        custom_words_list.add('แรงงาน')
        custom_words_list.add('จูราสสิคเวิลด์')
        custom_words_list.add('ไม่ใช่')
        custom_words_list.add('จู่โจม')
        custom_words_list.add('ไม่ให้')
        custom_words_list.add('กรมสวัสดิการและคุ้มครองแรงงาน')
        custom_words_list.add('กรมควบคุมโรค')
        custom_words_list.add('อนามัย')
        custom_words_list.add('บุหรี่ไฟฟ้า')
        custom_words_list.add('องค์กรพลังงาน')
        custom_words_list.add('แล้ว')
        custom_words_list.add('คริป')
        custom_words_list.add('คริปโต')
        custom_words_list.add('กระทรวงการพัฒนาสังคมและความมั่นคงของมนุษย์')
        custom_words_list.add('อันตราย')
        custom_words_list.add('องค์การค้า')
        custom_words_list.add('คริสไรท์')
        custom_words_list.add('ไม่น่า')
        custom_words_list.add('รูบิโอ')
        custom_words_list.add('เนทันยาฮู')
        custom_words_list.add('สีจิ้นผิง')
        custom_words_list.add('ทรีนิตี้')
        custom_words_list.add('เฟนทานิล')
        custom_words_list.add('อ่าว')
        custom_words_list.add('กวนตานาโม')
        custom_words_list.add('กล้าธรรม')
        custom_words_list.add('เซเลนสกี')
        custom_words_list.add('ดีอี')
        custom_words_list.add('ออนแทรีโอ')
        custom_words_list.add('สมศักดิ์')
        custom_words_list.add('อิชิบะ')
        custom_words_list.add('บิ๊กอ้วน')
        custom_words_list.add('คณะกรรมการกิจการกระจายเสียงกิจการโทรทัศน์และกิจการโทรคมนาคมแห่งชาติ')
        
        trie = dict_trie(dict_source=custom_words_list)
        _tokenizer = Tokenizer(custom_dict=trie, engine='newmm')

        list_data = []
        for i in range(len(data_list)): 
            word = _tokenizer.word_tokenize((data_list[i]))
            word = " ".join(word)
            list_data.append(word)
        return list_data
    
    def Stopwords(self, data_list): 
        cleaned_data_list = []
        for data in data_list:
            words = " ".join(word for word in data.split() if word not in thai_stopwords())
            cleaned_data_list.append(words)
        return cleaned_data_list

    def Lowercase(self, data_list):
        return [text.lower() for text in data_list]
        
NLP = NeuroLinguisticProgramming()

list_fake_news = NLP.load_data('fake_news.txt')
list_true_news = NLP.load_data('true_news.txt')

start_row = 500 
end_row = 520
text = NLP.Lemmatization(list_true_news[start_row:end_row]) #แปลงคำย่อ
text = NLP.Remove_Special_character(text) #ตัดอักขระพิเศษออก
text = NLP.Tokenization(text) #การสับคำ
# text = NLP.Stopwords(text) #การตัดคำที่ไม่สำคัญ
text = NLP.Lowercase(text) #เปลี่ยนพิมพ์ใหญ่เป็นพิมพ์เล็ก

count = start_row
for data in text: 
    print(f"{count}.) : {data}\n")
    count += 1
# 🛒 E-commerce Company


 <img src="https://user-images.githubusercontent.com/101379141/201035143-6f1af4fe-4169-4074-8287-6790d88803db.png" alt="Image" width="350" height="160">  



# :books: Table of Contents <!-- omit in toc -->

- [:briefcase: Business Case and Requirement](#briefcase-business-case-and-requirement)
- [:bookmark_tabs:Example Datasets](#bookmark_tabsexample-datasets)
- [A. Data Exploration and Cleansing](#a-data-exploration-and-cleansing)
- [B. Analysis](#b-analysis)

- [📃 What can you practice with this case study?](#what-can-you-practice-with-this-case-study)

---

# :briefcase: Business Case and Requirement


You are a Data Analyst working for an e-commerce company named X. You are tasked with preparing a presentation to present an overview of the company's business and operations to date for Sales and Operations Managers. 

❔ The presentation should include at a minimum the following information: 
- Business overview. 
- Customer satisfaction.  
- 2 to 3 areas of recommendation (areas) where the company can improve.

➕ Some additional information for the case study:
- Since there is only data up to 2018, we will assume that it is currently September 2018 (data after September 2018 you can ignore).
- The company is based in the US, but incorporated in Brazil (that's why some information is written in Portuguese).


---

# :bookmark_tabs:Example Datasets

### ✔ Orders dataset
Provide information about orders
- order_id: unique ID of the order
- customer_id: unique ID of the customer
- order_status: order status
- order_purchase_timestamp: time when the order was ordered
- order_approved_at: time the order is approved
- order_delivered_carrier_date: the time the item was delivered to the carrier
- order_delivered_customer_date: the time the item was delivered to the customer
- order_estimated_delivery_date: the estimated time the order will be delivered to the customer

<details><summary> 👆🏼 Click to expand Orders Dataset </summary>

<div align="center">

**Table: orders_dataset** 

<div align="center">
First 10 rows

|order_id|customer_id|order_status|order_purchase_timestamp|order_approved_at|order_delivered_carrier_date|order_delivered_customer_date|order_estimated_delivery_date|
|:----|:-----|:----|:----|:----|:----|:----|:----|
e481f51cbdc54678b7cc49136f2d6af7|	9ef432eb6251297304e76186b10a928d|	delivered|	10/2/2017 10:56|10/2/2017 11:07|	10/4/2017 19:55|	10/10/2017 21:25|	10/18/2017|
53cdb2fc8bc7dce0b6741e2150273451|	b0830fb4747a6c6d20dea0b8c802d7ef|	delivered|	7/24/2018 20:41|	7/26/2018 3:24|	7/26/2018 14:31|	8/7/2018 15:27|	8/13/2018|
47770eb9100c2d0c44946d9cf07ec65d|	41ce2a54c0b03bf3443c3d931a367089|	delivered|	8/8/2018 8:38|	8/8/2018 8:55|	8/8/2018 13:50|	8/17/2018 18:06|	9/4/2018|
949d5b44dbf5de918fe9c16f97b45f8a|	f88197465ea7920adcdbec7375364d82|	delivered|	11/18/2017 19:28|	11/18/2017 19:45|	11/22/2017 13:39|	12/2/2017 0:28|	12/15/2017|
ad21c59c0840e6cb83a9ceb5573f8159|	8ab97904e6daea8866dbdbc4fb7aad2c|	delivered|	2/13/2018 21:18|	2/13/2018 22:20|	2/14/2018 19:46|	2/16/2018 18:17	|2/26/2018|
a4591c265e18cb1dcee52889e2d8acc3|	503740e9ca751ccdda7ba28e9ab8f608|	delivered|	7/9/2017 21:57|	7/9/2017 22:10|	7/11/2017 14:58|	7/26/2017 10:57|	8/1/2017|
136cce7faa42fdb2cefd53fdc79a6098|	ed0271e0b7da060a393796590e7b737a|	invoiced|	4/11/2017 12:22|	4/13/2017 13:25|||			|5/9/2017|
6514b8ad8028c9f2cc2374ded245783f|	9bdf08b4b3b52b5526ff42d37d47f222|	delivered|	5/16/2017 13:10|	5/16/2017 13:22|	5/22/2017 10:07|	5/26/2017 12:55|	6/7/2017|
76c6e866289321a7c93b82b54852dc33|	f54a9f0e6b351c431402b8461ea51999|	delivered|	1/23/2017 18:29|	1/25/2017 2:50|	1/26/2017 14:16|	2/2/2017 14:08|	3/6/2017|
e69bfb5eb88e0ed6a785585b27e16dbf|	31ad1d1b63eb9962463f764d4e6e0c9d|	delivered|	7/29/2017 11:55|	7/29/2017 12:05|	8/10/2017 19:45|	8/16/2017 17:14|	8/23/2017|

</div>
</div>

</details>

---

### ✔ Order items dataset  
Provide information about each item in the order and shipping costs
- order_id: unique ID of the order
- order_item_id: ID of the item in the order (item number 1 has ID 1, item 2 has ID 2, etc. Based on this we also know how many items each order has)
- product_id: unique ID of the product in the order
- seller_id: unique ID of the seller
- price: the price of the item
- freight_value: shipping fee

<details><summary> 👆 Click to expand Orders Items Dataset </summary>

<div align="center">

**Table: order_items_dataset** 

<div align="center">
First 10 rows


|order_id|order_item_id|product_id|seller_id|price|freight_value|
|:----|:-----|:----|:----|:----|:----|
00010242fe8c5a6d1ba2dd792cb16214|	1|	4244733e06e7ecb4970a6e2683c13e61|	48436dade18ac8b2bce089ec2a041202|	58.9|	13.29|
00018f77f2f0320c557190d7a144bdd3|	1|	e5f2d52b802189ee658865ca93d83a8f|dd7ddc04e1b6c2c614352b383efe2d36|	239.9|	19.93|
000229ec398224ef6ca0657da4fc703e|	1|	c777355d18b72b67abbeef9df44fd0fd|	5b51032eddd242adc84c38acab88f23d|	199|	17.87|
00024acbcdf0a6daa1e931b038114c75|	1|	7634da152a4610f1595efa32f14722fc|	9d7a1d34a5052409006425275ba1c2b4|	12.99|	12.79|
00042b26cf59d7ce69dfabb4e55b4fd9|	1|	ac6c3623068f30de03045865e4e10089|	df560393f3a51e74553ab94004ba5c87|	199.9|	18.14|
00048cc3ae777c65dbb7d2a0634bc1ea|	1	|ef92defde845ab8450f9d70c526ef70f|	6426d21aca402a131fc0a5d0960a3c90|	21.9|	12.69|
00054e8431b9d7675808bcb819fb4a32|	1|	8d4f2bb7e93e6710a28f34fa83ee7d28|	7040e82f899a04d1b434b795a43b4617|	19.9|	11.85|
000576fe39319847cbb9d288c5617fa6|	1|	557d850972a7d6f792fd18ae1400d9b6|	5996cddab893a4652a15592fb58ab8db|	810|	70.75|
0005a1a1728c9d785b8e2b08b904576c|	1|	310ae3c140ff94b03219ad0adc3c778f|	a416b6a846a11724393025641d4edd5e|	145.95|	11.65|
0005f50442cb953dcd1d21e1fb923495|	1|	4535b0e1091c278dfd193e5a1d63b39f|	ba143b05f0110f0dc71ad71b4466ce92|	53.99|	11.4|
  
</div>
</div>

</details>

---

### ✔ Order payments dataset
Provide information of order payments.

*Note that we need to combine all values of each order to have total values.*

- order_id: unique ID of order
- payment_sequential: sequence order
- payment_type: payment type
- payment_installments: full payment (payment_installments = 1) or installment (payment_installments > 1,total payment is splited to many payments .
- payment_value: payment value (payment_value - equal total payments of all times payment installments)

<details><summary> 👆 Click to expand Orders Payments Dataset </summary>

<div align="center">

**Table: order_payments_dataset** 

<div align="center">
First 10 rows

|order_id|payment_sequential|payment_type|payment_installments|payment_value|
|:----|:-----|:----|:----|:----|
b81ef226f3fe1789b1e8b2acac839d17| 1	|credit_card|	8|	99.33|
a9810da82917af2d9aefd1278f1dcfa0|	1	|credit_card|	1|	24.39|
25e8ea4e93396b6fa0d3dd708e76c1bd|	1	|credit_card|	1|	65.71|
ba78997921bbcdc1373bb41e913ab953|	1	|credit_card|	8|	107.78|
42fdf880ba16b47b59251dd489d4441a|	1	|credit_card|	2|	128.45|
298fcdf1f73eb413e4d26d01b25bc1cd|	1	|credit_card|	2|	96.12|
771ee386b001f06208a7419e4fc1bbd7|	1	|credit_card|	1|	81.16|
3d7239c394a212faae122962df514ac7|	1	|credit_card|	3|	51.84|
1f78449c87a54faf9e96e88ba1491fa9|	1	|credit_card|	6|	341.09|
0573b5e23cbd798006520e1d5b4c6714|	1	|cash|	1|	51.95|


</div>
</div>

</details>

---

### ✔ Product dataset 
Provide product information
- product_id: unique ID of product
- product_category_name: category product name 
- product_name_lenght: number of product name letters
- product_description_lenght: number of product description letters
- product_photos_qty: number of product photo
- product_weight_g: weight of product  (g)
- product_length_cm: length of product (cm)
- product_height_cm: height of product (cm)
- product_width_cm: width/deep of product (cm)

<details><summary> 👆 Click to expand Product Dataset </summary>

<div align="center">

**Table: products_dataset** 

<div align="center">
First 10 rows

|product_id|product_category_name|product_name_lenght|product_description_lenght|product_photos_qty|product_weight_g|product_length_cm|product_height_cm|product_width_cm|
|:----|:-----|:----|:----|:----|:----|:----|:----|:----|
1e9e8ef04dbcff4541ed26657ea517e5|perfumaria|40|287|1|225|16|10|14|
3aa071139cb16b67ca9e5dea641aaa2f|artes|44|276|1|1000|30|18|20|
96bd76ec8810374ed1b65e291975717f|esporte_lazer|	46|	250|	1	|154|	18|	9|	15|
cef67bcfe19066a932b7673e239eb23d|bebes|	27|	261|	1|	371|	26|	4|	26|
9dc1a7de274444849c219cff195d0b71|utilidades_domesticas|	37|	402|	4|	625|	20|	17|	13|
41d3672d4792049fa1779bb35283ed13|instrumentos_musicais|	60|	745|	1|	200|	38|	5|	11|
732bd381ad09e530fe0a5f457d81becb|cool_stuff|	56|	1272|	4|	18350|	70|	24|	44|
2548af3e6e77a690cf3eb6368e9ab61e|moveis_decoracao|	56|	184|	2|	900|	40|	8|	40|
37cc742be07708b53a98702e77a21a02|eletrodomesticos|	57|	163	|1	|400	|27	|13	|17|
8c92109888e8cdf9d66dc7e463025574|brinquedos|	36|	1156|	1|	600|	17|	10|	12|


</div>
</div>

</details>

---

### ✔ Product category name translation
Translate the product name from Portuguese to English

- product_category_name
- product_category_name_english

<details><summary> 👆 Click to expand Product Category Name Translation Dataset </summary>


<div align="center">

**Table: product_category_name_translation** 

<div align="center">
First 10 rows

|product_category_name|product_category_name_english|
|:----|:-----|
beleza_saude|health_beauty|
informatica_acessorios|computers_accessories|
automotivo	|auto|
cama_mesa_banho	|bed_bath_table|
moveis_decoracao	|furniture_decor|
esporte_lazer	|sports_leisure|
perfumaria	|perfumery|
utilidades_domesticas|	housewares|
telefonia|	telephony|
relogios_presentes|watches_gifts|

</div>
</div>

</details>

---

### ✔ Order reviews dataset 
Provide review details of each order
- review_id: unique ID of revie
- order_id: unique ID of order
- review_score: Review Score
- review_comment_title: Comment title
- review_comment_message: detail of review
- review_creation_date: Created date of review
- review_answer_timestamp: timestamp of review answers

<details><summary>  Click to expand Order Reviews Dataset </summary>

<div align="center">

**Table: order_reviews_dataset** 

<div align="center">
First 10 rows

|review_id|order_id|review_score|review_comment_title|review_comment_message|review_creation_date|review_answer_timestamp|
|:----|:-----|:----|:----|:----|:----|:----|
7bc2406110b926393aa56f80a40eba40|73fc7af87114b39712e6da79b0a377eb|4|		|	|1/18/2018 0:00|	1/18/2018 21:46|
80e641a11e56f04c1ad469d5645fdfde|a548910a1c6147796b98fdf73dbeba33|5|			3/10/2018 0:00|	3/11/2018 3:05|
228ce5500dc1d8e020d8d1322874b6f0|f9e4b658b201a9f2ecdecbb34bed034b|5|			2/17/2018 0:00|	2/18/2018 14:36|
e64fb393e7b32834bb789ff8bb30750e|658677c97b385a9be170737859d3511b|5|		|Recebi bem antes do prazo estipulado.|	4/21/2017 0:00|	4/21/2017 22:02|
f7c4243c7fe1938f181bec41a392bdeb|8e6bfb81e283fa7e4f11123a3fb894f1|5|		|ParabÃ©ns lojas lannister adorei comprar pela Internet seguro e prÃ¡tico ParabÃ©ns a todos feliz PÃ¡scoa|	3/1/2018 0:00|	3/2/2018 10:26|
15197aa66ff4d0650b5434f1b46cda19|b18dcdf73be66366873cd26c5724d1dc	|1|		|	|4/13/2018 0:00	|4/16/2018 0:39|
07f9bee5d1b850860defd761afa7ff16|e48aa0d2dcec3a2e87348811bcfdf22b	|5|		|	|7/16/2017 0:00	|7/18/2017 19:30|
7c6400515c67679fbee952a7525281ef|c31a859e34e3adac22f376954e19b39d|5	|		| |8/14/2018 0:00	|8/14/2018 21:36|
a3f6f7f6f433de0aefbb97da197c554c|9c214ac970e84273583ab523dfafd09b|5|			| |5/17/2017 0:00	|5/18/2017 12:05|
8670d52e15e00043ae7de4c01cc2fe06|b9bf720beb4ab3728760088589c62129|4|	recomendo|	aparelho eficiente. no site a marca do aparelho esta impresso como 3desinfector e ao chegar esta com outro nome...atualizar com a marca correta uma vez que Ã© o mesmo aparelho|	5/22/2018 0:00|	5/23/2018 16:45|



</div>
</div>

</details>

---
### ✔ Customers dataset
Provide Customer Information 

- customer_id: customer unique ID ( used to link with customer_id of orders_dataset table.
- customer_unique_id: mã unique ID of customer in system of customer information management. 
- customer_zip_code_prefix: zip code of customer
- customer_city: City name of customer 
- customer_state: State name of customer

<details><summary> 👆 Click to expand Customers Dataset </summary>

<div align="center">

**Table: Customers_dataset** 
 
<div align="center">
First 10 rows

|customer_id|customer_unique_id|customer_zip_code_prefix|customer_city|customer_state|
|:----|:-----|:----|:----|:----|
06b8999e2fba1a1fbc88172c00ba8bc7|861eff4711a542e4b93843c6dd7febb0|	14409|	franca	|SP|
18955e83d337fd6b2def6b18a428ac77|	290c77bc529b7ac935b93aa66c333dc3|	9790|	sao bernardo do campo|	SP|
4e7b3e00288586ebd08712fdd0374a03|	060e732b5b29e8181a18229c7b0b2b5e|	1151|	sao paulo	|SP|
b2b6027bc5c5109e529d4dc6358b12c3|	259dac757896d24d7702b9acbbff3f3c|	8775|	mogi das cruzes|	SP|
4f2d8ab171c80ec8364f7c12e35b23ad|	345ecd01c38d18a9036ed96c73b8d066|	13056|	campinas	|SP|
879864dab9bc3047522c92c82e1212b8|	4c93744516667ad3b8f1fb645a3116a4|	89254|	jaragua do sul	|SC|
fd826e7cf63160e536e0908c76c3f441|	addec96d2e059c80c30fe6871d30d177|	4534|	sao paulo	|SP|
5e274e7a0c3809e14aba7ad5aae0d407|	57b2a98a409812fe9618067b6b8ebe4f|	35182|	timoteo	|MG|
5adf08e34b2e993982a47070956c5c65|	1175e95fb47ddff9de6b2b06188f7e0d|	81560|	curitiba	|PR|
4b7139f34592b3a31687243a302fa75b|	9afe194fb833f79e300e37e580171f22|	30575|	belo horizonte|	MG|


</div>
</div>

</details>

---


# A. [Data Exploration and Cleansing](https://github.com/beto1810/E-commerce-Company/blob/main/A.Data%20Exploration%20%26%20Cleansing.md)



# B. [Analysis](https://github.com/beto1810/E-commerce-Company/blob/main/B.Analysis.md)


---

# 🧾 What can you practice with this case study?
- Python
  - pandas, numpy,matplotlib,seaborn.
  - cleaning, check Null values, transforming.
 
  - import, save csv file. 
- POWER BI
  - Visualize
  - Analyze
  - Measure, DAX, etc
  - Import CSV File and Transform data

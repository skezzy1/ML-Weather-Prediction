import pandas as pd

dataset = pd.read_csv("data_preparation/csv/data.csv")
region_map = {
    # Захід
    "Володимир": "West", "Сарни":"West", "Хмельницький":"West", "Шепетівка": "West",
    "Львів": "West", "Івано-Франківськ": "West", "Ужгород": "West", "Тернопіль": "West",
    "Рівне": "West", "Дрогобич": "West", "Коломия": "West", "Стрий": "West",
    "Яворів": "West", "Золочів": "West", "Славське": "West", "Турка": "West",
    "Мостиська": "West", "Рахів": "West", "Берегове": "West", "Чортків": "West",
    "Бережани":"West", "Броди":"West", "Дружба":"West", "Кам’янець-Подільський": "West",
    "Ковель":"West", "Крем’янець":"West", "Луцьк":"West", "Нова Ушиця":"West",
    "Рава Руська":"West", "Турка":"West", "Хуст":"West", "Світязь":"West",
    "Великий Березний":"West", "Долина":"West", "Любешів":"West", "Нижні Ворота":"West",
    "Плай":"West", "Яремча":"West", "Дубно":"West", "Міжгір’я":"West", "Маневичі":"West",
    "Нижній Студений":"West", "Пожижевська":"West", "Стрий":"West", "Золочів":"West",
    "Мукачево":"West", "Костопіль":"West",
    # Північ
    "Київ": "North", "Житомир": "North", "Чернігів": "North", "Суми": "North",
    "Коростень": "North", "Ніжин": "North", "Сновськ": "North", "Овруч": "North",
    "Чорнобиль": "North", "Остер": "North", "Конотоп": "North", "Миронівка":"North",
    "Білопілля":"North", "Звягель":"North", "Олевськ":"North", "Прилуки":"North",
    "Ромни":"North", "Семенівка":"North", "Тетерів": "North", "Бориспіль":"North",
    "Глухів":"North", "Лебедин":"North", "Баришівка":"North", "Покошичі":"North",
    # Схід
    "Маріуполь":"East", "Богодухів":"East", "Великий Бурлук":"East",
    "Харків": "East", "Дніпро": "East", "Запоріжжя": "East", "Луганськ": "East",
    "Донецьк": "East", "Ізюм": "East", "Сватове": "East", "Бахмут": "East",
    "Куп’янськ": "East", "Покровськ": "East", "Павлоград": "East", "Губиниха":"East",
    "Комісарівка":"East","Красноград":"East", "Лозова": "East", "Пришиб":"East",
    "Чаплине":"East", "Дебальцеве":"East", "Коломак":"East", "Біловодськ":"East",
    "Нікополь":"East", "Синельникове":"East", "Слобожанське":"East", "Слов’янськ":"East",
    # Південь
    "Керч":"South", "Любашівка":"South","Чернівці":"South", "Чорноморське":"South",
    "Асканія Нова":"South", "Велика Олександрівка":"South", "Бехтери":"South",
    "Одеса": "South", "Миколаїв": "South", "Херсон": "South", "Сімферополь": "South",
    "Феодосія": "South", "Євпаторія": "South", "Ялта": "South", "Алушта": "South",
    "Керч": "South", "Севастополь": "South", "Ізмаїл": "South", "Чорноморськ": "South",
    "Генічеськ": "South", "Кирилівка": "South", "Очаків": "South", "Хорли": "South",
    "Вознесенськ":"South", "Клепиніне":"South", "Нова Каховка":"South", "Селятин":"South",
    "Феодосія":"South", "Гурзуф":"South", "Ай-Петрі":"South","Баштанка":"South", "Ботієве":"South",
    "Волноваха":"South", "Гуляй Поле":"South","Мелітополь":"South", "Нижні Сірогози":"South",
    "Первомайськ":"South", "Роздільна":"South", "Сарата":"South", "Сербка":"South", 
    "Білгород-Дністровський":"South", "Болград":"South", "Вилкове":"South", "Затишшя":"South",
    "Новодністровськ":"South", "Бердянськ":"South","Енергодар":"South",
    # Центр
    "Вінниця": "Center", "Лубни":"Center", "Могилів-Подільський":"Center",
    "Полтава": "Center", "Черкаси": "Center", "Кропивницький": "Center",
    "Кривий Ріг": "Center", "Умань": "Center", "Світловодськ": "Center",
    "Сміла": "Center", "Гайворон": "Center", "Знам’янка": "Center",
    "Біла Церква": "Center", "Фастів": "Center", "Яготин": "Center",
    "Бобринець":"Center", "Веселий Поділ":"Center", "Гадяч":"Center",
    "Гайсин":"Center", "Жашків":"Center", "Жмеринка":"Center", "Звенигородка":"Center",
    "Кобеляки":"Center", "Новомиргород":"Center", "Хмільник":"Center", "Чигирин":"Center",
    "Золотоноша":"Center", "Помічна":"Center", "Ямпіль":"Center", "Долинська":"Center",
    "Канів":"Center", "Кременчук":"Center", "Тетерів (Пісківка)":"Center",
}
def region_exchanger(region_name):
    region_mapping = {
        "West": 0,
        "North": 1,
        "East": 2,
        "South": 3,
        "Center": 4
    }
    return region_mapping.get(region_name, -1)

dataset["Region"] = dataset["City"].map(region_map).apply(region_exchanger)
dataset['Average Wind Speed (m/s)'] = dataset['Average Wind Speed (m/s)'].fillna(
    dataset['Average Wind Speed (m/s)'].mean()
)

dataset.to_csv("data_preparation/csv/data_with_regions.csv", index=False)
print("✅ Колонку 'Region' додано та збережено у data_with_regions.csv")
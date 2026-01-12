

import re
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from colorama import init, Fore, Style
init(autoreset=True)


# -------------------------------------------------------
# Набор данных для тренировки
# -------------------------------------------------------
phishing_messages = [
    "Ваш аккаунт заблокирован, перейдите по ссылке и подтвердите данные",
    "Срочно! Ваша карта будет заморожена, войдите на сайт и введите код",
    "Вы выиграли приз! Оплатите доставку по ссылке",
    "Мы заметили подозрительный вход, подтвердите личность",
    "Уважаемый клиент! Ваш счёт заблокирован. Немедленно перейдите по http://bank-secure-login.ru и подтвердите данные, иначе доступ будет закрыт!",
    "Срочно! Вы выиграли приз. Для получения перевода укажите номер карты и CVV на http://get-prize.example",
    "Ваш аккаунт нарушил правила. Подтвердите личность: пришлите паспортные данные",
    "Внимание! Мы обнаружили подозрительную активность. Подтвердите пароль на http://secure-login-example.com",
    "Поздравляем! Вы получаете компенсацию. Переведите 1000 руб для активации",
    "Уважаемый абонент! Ваша SIM-карта будет деактивирована.\nДля сохранения номера отправьте паспортные данные на email...",
]

legit_messages = [
    "Уроки на следующую неделю выложены в электронный журнал",
    "Напоминаем о собрании родителей в пятницу",
    "Уведомление: расписание уроков на следующую неделю обновлено. Посмотреть можно в электронном журнале.",
    "Привет! Это учитель класса. Не забудьте завтра принести лабораторную работу.",
    "Информация от школы: собрание родителей пройдет в пятницу в 18:00.",
    "Сообщение от администрации: технические работы в воскресенье с 2:00 до 4:00.",
]

neutral_messages = [
    "Привет, как дела?",
    "Добрый день!",
    "Яблоки сегодня очень свежие",
    "Погода сегодня прекрасная",
    "Я тебя люблю <3"
    "Привет путник из мира грёз!! xD"
]

texts = phishing_messages + legit_messages + neutral_messages
labels = [1] * len(phishing_messages) + [0] * len(legit_messages) + [0] * len(neutral_messages)

# -------------------------------------------------------
# Ключевые опасные словечки!!! >:3
# -------------------------------------------------------
URGENCY_WORDS = ['сроч', 'немедленно', 'иначе', 'срочно', 'последн', 'быстр', 'заблокир', 'заморож', 'удал', 'деактив']
MONEY_WORDS = ['перевед', 'оплат', 'карт', 'cvv', 'cvc', 'руб', 'выплат', 'штраф', 'плат', 'деньг', 'приз', 'компенс']
PERSONAL_WORDS = ['паспорт', 'пароль', 'cvv', 'cvc', 'паспортные', 'данные', 'подтверд', 'счет', 'счёт', 'email',
                  'почт', 'логин']
SUSPICIOUS_DOM_PATTERNS = ['-', 'login', 'secure', 'verify', 'bank-', 'account', 'update']


# -------------------------------------------------------
# 3. Расчет признаков
# -------------------------------------------------------
def simple_preprocess(text):
    return text.lower().replace('\n', ' ').strip()


def count_urls(text):
    return len(re.findall(r'http[s]?://', text.lower()))


def caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def contains_any(text, keywords):
    t = text.lower()
    return any(word in t for word in keywords)


def suspicious_domain_indicator(text):
    matches = re.findall(r'http[s]?://([A-Za-z0-9\-\._]*)', text.lower())
    if not matches:
        return 0
    for dom in matches:
        for p in SUSPICIOUS_DOM_PATTERNS:
            if p in dom:
                return 1
    return 0


def extract_rule_features(text):
    features = np.array([
        1 if contains_any(text, URGENCY_WORDS) else 0,
        1 if contains_any(text, MONEY_WORDS) else 0,
        1 if contains_any(text, PERSONAL_WORDS) else 0,
        count_urls(text),
        caps_ratio(text),
        suspicious_domain_indicator(text)
    ], dtype=float)

    features[3] = min(features[3], 3) / 3.0
    features[4] = min(features[4], 0.5) * 2

    return features


def get_rule_score(text):
    features = extract_rule_features(text)
    weights = [30, 25, 30, 15, 10, 40]
    score = np.dot(features, weights)
    return min(score, 100)


# -------------------------------------------------------
# 4. Обучение модели
# -------------------------------------------------------
print("[*] Загрузка системы безопасности...")

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=300)
X_text = vectorizer.fit_transform([simple_preprocess(t) for t in texts])

X_num = np.vstack([extract_rule_features(t) for t in texts])
X_num_sparse = csr_matrix(X_num)

X = hstack([X_text, X_num_sparse])
y = np.array(labels)

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("[OK] Система готова к работе!\n")


# -------------------------------------------------------
# 5. Функция анализа
# -------------------------------------------------------
def analyze_message(text):
    t = simple_preprocess(text)

    x_text = vectorizer.transform([t])
    x_num = csr_matrix(extract_rule_features(t).reshape(1, -1))
    x = hstack([x_text, x_num])

    ml_prob = float(model.predict_proba(x)[0, 1]) * 100
    rule_score = get_rule_score(text)

    final_score = 0.6 * ml_prob + 0.4 * rule_score

    rule_features = extract_rule_features(text)
    if rule_score < 10 and ml_prob < 20:
        final_score = 0

    if count_urls(t) > 0 and suspicious_domain_indicator(t):
        final_score = max(final_score, 85)

    explanations = []
    if rule_features[0] > 0:
        explanations.append("[!] Слова срочности/давления")
    if rule_features[1] > 0:
        explanations.append("[!] Упоминание денег/платежей")
    if rule_features[2] > 0:
        explanations.append("[!] Запрос личных данных")
    if rule_features[3] > 0:
        explanations.append(f"[!] Ссылки: {int(rule_features[3] * 3)} шт")
    if rule_features[4] > 0.3:
        explanations.append("[!] Много заглавных букв")
    if rule_features[5] > 0:
        explanations.append("[!] Подозрительный домен")

    if final_score >= 70:
        recommendation = Fore.RED + "[!] ВЫСОКИЙ РИСК! НЕ переходите по ссылкам!"
    elif final_score >= 40:
        recommendation = Fore.YELLOW + "[0-0] СРЕДНИЙ РИСК. Будьте осторожны."
    elif final_score >= 20:
        recommendation = Fore.GREEN + "[OK] НИЗКИЙ РИСК. Проверьте детали."
    else:
        recommendation = Fore.GREEN + "[OK] БЕЗОПАСНО."

    print("\n" + "=" * 50)
    print("[*] РЕЗУЛЬТАТ АНАЛИЗА")
    print("=" * 50)
    print(f"[*] Вероятность фишинга: {final_score:.1f}%")

    if explanations:
        print("\n[!] Обнаружены признаки:")
        for exp in explanations[:3]:
            print(f"   • {exp}")

    print("\n" + recommendation)

    print("=" * 50 + "\n")

    return final_score


# -------------------------------------------------------
# 6. Ввод (МНОГОСТРОЧНЫЙ ОУУ ЕС)
# -------------------------------------------------------
def get_multiline_input():
    print("\n[*] Введите сообщение (для завершения - Enter на пустой строке):")
    print("   Или введите 'выход' для завершения работы")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "":
                if lines:
                    break
                else:
                    continue
            if line.lower() in ['выход', 'exit', 'quit']:
                return None
            lines.append(line)
        except EOFError:
            break

    if not lines:
        return None

    return "\n".join(lines)


# -------------------------------------------------------
# 7. Главный цикл программы
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("[0-0] PHISHGUARD V0.2 - Детектор фишинговых сообщений [0-0]")
    print("=" * 60)
    print("Консольная программа. Система анализирует текст и определяет риск фишинга.")
    print("Введите сообщение для проверки безопасности.\n")

    while True:
        user_input = get_multiline_input()

        if user_input is None:
            print("\n [0-0] Спасибо за использование!")
            print("[*] Будьте осторожны в интернете!")
            break

        analyze_message(user_input)


# -------------------------------------------------------
# 8. ЗАПУСК (стартуееем)
# -------------------------------------------------------
if __name__ == "__main__":
    main()
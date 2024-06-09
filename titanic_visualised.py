import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データを読み込む
train_df = pd.read_csv('/Users/madoka/trantura_ondisk/practice/train.csv')

# 警告を避けるための欠損値補完
train_df['Fare'] = train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())

# インタラクティブモードを有効にする
plt.ion()

# グラフ1: 'Age' のヒストグラム
print("Age Distribution plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()  # グラフを表示

# グラフ2: 'Survived' のカウントプロット
print("Survival Count plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.countplot(x='Survived', data=train_df)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()  # グラフを表示

# グラフ3: 'Pclass' ごとの生存者数のカウントプロット
print("Survival Count by Pclass plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.countplot(x='Pclass', hue='Survived', data=train_df)
plt.title('Survival Count by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()  # グラフを表示

# グラフ4: 'Fare' のボックスプロット
print("Fare Distribution by Pclass plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.boxplot(x='Pclass', y='Fare', data=train_df)
plt.title('Fare Distribution by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Fare')
plt.show()  # グラフを表示

# グラフ5: 'Embarked' ごとの生存者数のカウントプロット
print("Survival Count by Embarked plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.countplot(x='Embarked', hue='Survived', data=train_df)
plt.title('Survival Count by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.show()  # グラフを表示

# 家族のサイズを計算
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1  # +1 は本人を含む

# グラフ6: 家族サイズの分布
print("Family Size Distribution plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.barplot(x=train_df['FamilySize'].value_counts().index, y=train_df['FamilySize'].value_counts().values, palette='viridis')
plt.title('Family Size Distribution')
plt.xlabel('Family Size')
plt.ylabel('Number of Passengers')
plt.show()  # グラフを表示

# 家族がいるかどうかでグループ化して生存率をプロット
train_df['HasFamily'] = train_df['FamilySize'].apply(lambda x: 'Alone' if x == 1 else 'With Family')

# グラフ7: 家族の有無による生存率
print("Survival Rate by Family Presence plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.barplot(x=train_df.groupby('HasFamily')['Survived'].mean().index, y=train_df.groupby('HasFamily')['Survived'].mean().values, palette='viridis')
plt.title('Survival Rate by Family Presence')
plt.xlabel('Family Presence')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()  # グラフを表示

# グラフ8: ファミリーサイズと生存率の関係をプロット
print("Survival Rate by Family Size plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.barplot(x=train_df.groupby('FamilySize')['Survived'].mean().index, y=train_df.groupby('FamilySize')['Survived'].mean().values, palette='viridis')
plt.title('Survival Rate by Family Size')
plt.xlabel('Family Size')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()  # グラフを表示

# グラフ9: 性別による生存率の相関関係をプロット
print("Survival Rate by Sex plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.barplot(x='Sex', y='Survived', data=train_df, palette='viridis')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()  # グラフを表示

# 年齢と生存率の関係をプロット
print("Survival Rate by Age plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.kdeplot(x='Age', hue='Survived', data=train_df, fill=True, common_norm=False, palette='viridis', alpha=0.5, linewidth=2)
plt.title('Survival Rate by Age')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()  # グラフを表示

# チケットの代金と生存率の関係をプロット
print("Survival Rate by Fare plotting")
plt.figure(figsize=(10, 6))  # 新しいFigureを作成
sns.kdeplot(x='Fare', hue='Survived', data=train_df, fill=True, common_norm=False, palette='viridis', alpha=0.5, linewidth=2)
plt.title('Survival Rate by Fare')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.show()  # グラフを表示

# インタラクティブモードを終了する
plt.ioff()

# 最後に全てのプロットを表示して終了する
plt.show(block=True)
# 0. 학습된 리소스
======================
다음 3개 파일을 다운받아서 같은 경로에 위치시킨다.
```
https://s3.ap-northeast-2.amazonaws.com/bestend.co.kr/article_to_id.pickle
https://s3.ap-northeast-2.amazonaws.com/bestend.co.kr/best_loss.h5
https://s3.ap-northeast-2.amazonaws.com/bestend.co.kr/config.json

```


# 1. config.py 수정
======================
config.py 파일을 열어서 리소스 경로(카카오에서 제공한 데이터 폴더 경로)와
cache(중간생성파일)가 생성될 폴더를 지정한다. 
 ```
RESOURCE_DIR = ROOT_DIR + '/../res/'
CACHE_DIR = ROOT_DIR + '/../cache/' 
 ```
 
# 2. 문서 임베딩 생성
======================
문서의 임베딩을 doc2vec방식으로 생성한다.
기본 파라메터로 학습이 종료될때까지 기다린다.

자세한 파라메터는 소스코드 참조. 
```
PYTHONPATH=./ python3 article_embedding.py --train_dir <학습폴더>
```

# 3. 학습
======================
ranking 모델 학습

자세한 파라메터는 소스코드 참조.
```
PYTHONPATH=./ python3 train.py --article_embedding_path <문서임베딩경로> --train_dir <학습폴더>
```

# 4. 추천결과
======================
```
PYTHONPATH=./ python3 scoring.py --model_path <모델경로:best_loss.h5> --user_set <dev.users|test.users>
```

* 비고: 여러가지 모델을 테스트 하기 위해 코드를 여러번 들어내서 고치다보니 중구난방에 여기저기 하드코딩된 알수없는 값들이 흩어져있다.
아레나가 끝나고 시간이 나면 언젠가는 코드 정리를 해보아야겠다.  
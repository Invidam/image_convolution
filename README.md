# Image Convolution

병렬처리를 이용한 이미지 컨볼루션 프로그램.

한국기술교육대학교 24-1 멀티코어프로그래밍(CSE241(01)) 중간 과제.

## 과제 목표
- 이미지 컨볼루션을 병렬처리로 구현하여 기존 대비 성능을 개선한다.

## 구성원

- 박한수 (팀장): 프로그램 입출력 처리, 성능 계산 역할, 발표
- 박찬영 (팀원): 컨볼루션의 병럴 처리
- 이우탁 (팀원): 이미지 리소스 조사, PPT 제작

## 설치

### opencv
- opencv 설치 참고: https://aerocode.net/279
  - 환경변수 버전을  4.5.5로 설정함. (변수 명 등)
  - git clone시 현재 경로에 클론됨을 유의할 것 (환경변수 경로 설정간)
  - cmd에서만 버전 확인이 가능함. (다른 터미널에서는 안됨.)
- https://www.youtube.com/watch?v=fjq8eTuHnMM 을 추천하는데 도움받지 못함. (`choco install opencv`가 안됨.)

### 기타 참고
- https://medium.com/@su_bak/git-github-com-permission-denied-publickey-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95-76b0ab741c62
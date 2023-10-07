import streamlit as st
from modules import config
import os



if __name__ == "__main__":
    config.set_config()
    os.makedirs("/app/temp/image", exist_ok=True)
    os.makedirs("/app/models", exist_ok=True)

"""
# 2023 DaiS Lab BTS 실전문제연구팀 프로젝트

## 용접강관의 비파괴 검사 이미지에 대한 딥러닝 기반 결함 탐지 플랫폼

### 프로젝트 기간 : 2023.04 ~

### 팀장 : Morteza

### 팀원 : 이창현, 김선영, 김창영, 박하연, 정근오, 조현건, 최준혁

"""


# 20242R0136COSE41600

인공지능과자율주행자동차

실행 방법
프로그램을 실행하려면 명령어 인자(--data_dir, --scenario)를 통해 data_dir과 scenario를 입력해야 합니다.

예시:
python your_script.py --data_dir /path/to/data --scenario scenario_name
python your_script.py --data_dir /home/user/pointcloud_data --scenario scenario_01

--data_dir: 데이터가 저장된 상위 디렉토리 경로입니다. 예를 들어, /path/to/data는 여러 시나리오의 하위 디렉토리를 포함한 경로여야 합니다.

--scenario: 처리할 시나리오의 이름입니다. 각 시나리오는 data_dir 내의 하위 폴더로 존재하며, 해당 폴더 안에 .pcd 파일들이 있어야 합니다.

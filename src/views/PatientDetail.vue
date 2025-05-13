<template>
  <div class="patient-detail" v-loading="loading">
    <div class="page-header">
      <h2>Карточка пациента</h2>
      <el-button @click="goBack">
        <el-icon><Back /></el-icon>
        Назад к списку
      </el-button>
    </div>

    <template v-if="patient">
      <el-card class="patient-info">
        <el-descriptions title="Информация о пациенте" :column="3" border>
          <el-descriptions-item label="Фамилия">{{ patient.lastName }}</el-descriptions-item>
          <el-descriptions-item label="Имя">{{ patient.firstName }}</el-descriptions-item>
          <el-descriptions-item label="Отчество">{{ patient.middleName }}</el-descriptions-item>
          <el-descriptions-item label="Дата рождения">{{ formatDate(patient.birthDate) }}</el-descriptions-item>
          <el-descriptions-item label="Пол">{{ patient.gender === 'male' ? 'Мужской' : 'Женский' }}</el-descriptions-item>
          <el-descriptions-item label="Номер полиса">{{ patient.policyNumber }}</el-descriptions-item>
        </el-descriptions>
      </el-card>

      <div class="analyses-header">
        <h3>Архив анализов костного возраста</h3>
        <el-button type="primary" @click="showAddAnalysisDialog">
          <el-icon><Plus /></el-icon>
          Добавить рентген
        </el-button>
      </div>

      <el-card v-if="patient.analyses.length > 0" class="analyses-list">
        <el-table :data="sortedAnalyses" style="width: 100%">
          <el-table-column prop="date" label="Дата рентгена">
            <template #default="{ row }">
              {{ formatDate(row.date) }}
            </template>
          </el-table-column>
          <el-table-column prop="predictedAge" label="Предсказанный возраст">
            <template #default="{ row }">
              {{ row.predictedAge.toFixed(1) }} лет
            </template>
          </el-table-column>
          <el-table-column label="Действия" width="150">
            <template #default="{ row }">
              <el-button type="primary" size="small" @click="showAnalysisDetail(row)">
                Подробнее
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-empty
          v-else
          description="Нет данных об анализах"
      />
    </template>

    <el-empty
        v-else-if="!loading"
        description="Пациент не найден"
    />

    <!-- Модалка для добавления нового анализа -->
    <el-dialog
        v-model="addAnalysisDialogVisible"
        title="Добавление рентгена"
        width="500px"
    >
      <el-form
          ref="analysisFormRef"
          :model="analysisForm"
          :rules="analysisRules"
          label-position="top"
      >
        <el-form-item label="Дата рентгена" prop="date">
          <el-date-picker
              v-model="analysisForm.date"
              type="date"
              placeholder="Выберите дату"
              style="width: 100%"
              format="DD.MM.YYYY"
              value-format="YYYY-MM-DD"
          />
        </el-form-item>

        <el-form-item label="Загрузите изображение рентгена" prop="xrayImage">
          <el-upload
              class="upload-demo"
              drag
              action="#"
              :auto-upload="false"
              :on-change="handleFileChange"
              :limit="1"
              accept=".jpg,.jpeg,.png"
          >
            <el-icon><UploadFilled /></el-icon>
            <div class="el-upload__text">
              Перетащите файл сюда или <em>нажмите для выбора</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                Только файлы JPG/PNG, не более 10MB
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="addAnalysisDialogVisible = false">Отмена</el-button>
          <el-button type="primary" @click="submitAnalysisForm(analysisFormRef)">
            Отправить
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- Модалка для просмотра информации -->
    <el-dialog
        v-model="analysisDetailDialogVisible"
        title="Детали анализа"
        width="700px"
    >
      <div v-if="selectedAnalysis" class="analysis-detail">
        <el-row :gutter="20">
          <el-col :span="12">
            <div class="analysis-info">
              <p><strong>Дата рентгена:</strong> {{ formatDate(selectedAnalysis.date) }}</p>
              <p><strong>Предсказанный возраст:</strong> {{ selectedAnalysis.predictedAge.toFixed(1) }} лет</p>

              <el-form :model="notesForm">
                <el-form-item label="Примечания доктора">
                  <el-input
                      v-model="notesForm.doctorNotes"
                      type="textarea"
                      rows="4"
                      placeholder="Введите примечания"
                  />
                </el-form-item>
                <el-form-item>
                  <el-button type="primary" @click="saveNotes">Сохранить примечания</el-button>
                </el-form-item>
              </el-form>
            </div>
          </el-col>
          <el-col :span="12">
            <div class="xray-image">
              <img :src="selectedAnalysis.xrayImage" alt="Рентген" style="max-width: 100%;" />
            </div>
          </el-col>
        </el-row>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import type { FormInstance, FormRules, UploadFile } from 'element-plus';
import { ElMessage } from 'element-plus';
import { Back, Plus, UploadFilled } from '@element-plus/icons-vue';
import { usePatientsStore } from '@/stores/patients';
import { type Analysis } from '@/types';

const route = useRoute();
const router = useRouter();
const patientsStore = usePatientsStore();
const loading = ref(true);

const patientId = computed(() => Number(route.params.id));
const patient = computed(() => patientsStore.getPatientById(patientId.value));

const sortedAnalyses = computed(() => {
  if (!patient.value) return [];
  return [...patient.value.analyses].sort((a, b) =>
      new Date(b.date).getTime() - new Date(a.date).getTime()
  );
});

const analysisFormRef = ref<FormInstance>();
const addAnalysisDialogVisible = ref(false);
const analysisForm = ref({
  date: '',
  xrayImage: '',
  file: null as File | null
});

const analysisRules: FormRules = {
  date: [
    { required: true, message: 'Пожалуйста, выберите дату', trigger: 'change' }
  ],
  xrayImage: [
    { required: true, message: 'Пожалуйста, загрузите изображение', trigger: 'change' }
  ]
};

const analysisDetailDialogVisible = ref(false);
const selectedAnalysis = ref<Analysis | null>(null);
const notesForm = ref({
  doctorNotes: ''
});

const formatDate = (dateString: string): string => {
  const date = new Date(dateString);
  return date.toLocaleDateString('ru-RU');
};

const goBack = () => {
  router.push({ name: 'home' });
};

const showAddAnalysisDialog = () => {
  analysisForm.value = {
    date: new Date().toISOString().split('T')[0],
    xrayImage: '',
    file: null
  };
  addAnalysisDialogVisible.value = true;
};

const handleFileChange = (file: UploadFile) => {
  const isImage = file.raw?.type.startsWith('image/');
  const isLt10M = file.size / 1024 / 1024 < 10;

  if (!isImage) {
    ElMessage.error('Можно загружать только изображения!');
    return false;
  }

  if (!isLt10M) {
    ElMessage.error('Размер изображения не должен превышать 10MB!');
    return false;
  }

  analysisForm.value.xrayImage = URL.createObjectURL(file.raw!);
  analysisForm.value.file = file.raw;
  return true;
};

const submitAnalysisForm = async (formEl: FormInstance | undefined) => {
  if (!formEl) return;

  await formEl.validate((valid) => {
    if (valid && patient.value) {
      const predictedAge = Math.random() * 5 + 10;

      patientsStore.addAnalysis(patientId.value, {
        date: analysisForm.value.date,
        predictedAge,
        xrayImage: analysisForm.value.xrayImage,
        doctorNotes: ''
      });

      ElMessage({
        message: 'Анализ успешно добавлен',
        type: 'success'
      });

      addAnalysisDialogVisible.value = false;
    }
  });
};

const showAnalysisDetail = (analysis: Analysis) => {
  selectedAnalysis.value = analysis;
  notesForm.value.doctorNotes = analysis.doctorNotes || '';
  analysisDetailDialogVisible.value = true;
};

const saveNotes = () => {
  if (selectedAnalysis.value && patient.value) {
    patientsStore.updateAnalysisNotes(
        patientId.value,
        selectedAnalysis.value.id,
        notesForm.value.doctorNotes
    );

    ElMessage({
      message: 'Примечания сохранены',
      type: 'success'
    });

    const updatedAnalysis = patientsStore.getAnalysisByIds(
        patientId.value,
        selectedAnalysis.value.id
    );
    if (updatedAnalysis) {
      selectedAnalysis.value = updatedAnalysis;
    }
  }
};

onMounted(() => {
  setTimeout(() => {
    loading.value = false;
    if (!patient.value) {
      ElMessage({
        message: 'Пациент не найден',
        type: 'error'
      });
    }
  }, 500);
});
</script>

<style scoped>
.patient-detail {
  padding: 20px 0;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.patient-info {
  margin-bottom: 20px;
}

.analyses-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 20px 0;
}

.analyses-list {
  margin-bottom: 20px;
}

.analysis-detail {
  padding: 10px;
}

.xray-image {
  text-align: center;
}
</style>
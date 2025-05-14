<template>
  <div class="patient-detail" v-loading="loadingPatient">
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
          <el-descriptions-item label="Отчество">{{ patient.middleName || '-' }}</el-descriptions-item>
          <el-descriptions-item label="Дата рождения">{{ formatDate(patient.birthDate) }}</el-descriptions-item>
          <el-descriptions-item label="Пол">{{ patient.gender === 'male' ? 'Мужской' : 'Женский' }}</el-descriptions-item>
          <el-descriptions-item label="Номер полиса">{{ patient.policyNumber }}</el-descriptions-item>
        </el-descriptions>
      </el-card>

      <div class="analyses-header">
        <h3>Архив анализов костного возраста</h3>
        <el-button type="primary" @click="showAddAnalysisDialog" :disabled="!patient">
          <el-icon><Plus /></el-icon>
          Добавить рентген
        </el-button>
      </div>

      <el-card v-if="sortedAnalyses.length > 0" class="analyses-list">
        <el-table :data="sortedAnalyses" style="width: 100%">
          <el-table-column prop="date" label="Дата рентгена" min-width="120">
            <template #default="{ row }">
              {{ formatDate(row.date) }}
            </template>
          </el-table-column>
          <el-table-column prop="predictedAge" label="Предсказанный возраст" min-width="150">
            <template #default="{ row }">
              {{ row.predictedAge !== null && row.predictedAge !== undefined ? row.predictedAge.toFixed(1) + ' лет' : 'Обработка...' }}
            </template>
          </el-table-column>
          <el-table-column label="Действия" width="150" fixed="right">
            <template #default="{ row }">
              <el-button type="primary" size="small" @click="showAnalysisDetail(row)">
                Подробнее
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-empty
          v-else-if="!loadingPatient && patient"
          description="У этого пациента еще нет анализов. Добавьте первый."
      />
    </template>

    <el-empty
        v-else-if="!loadingPatient && !patient"
        description="Пациент не найден. Возможно, он был удален или ссылка неверна."
    />

    <!-- Модалка для добавления нового анализа -->
    <el-dialog
        v-model="addAnalysisDialogVisible"
        title="Добавление рентгена"
        width="500px"
        :close-on-click-modal="false"
        @closed="resetAnalysisFormDialog"
    >
      <el-form
          ref="analysisFormRef"
          :model="analysisForm"
          :rules="analysisRules"
          label-position="top"
          @submit.prevent="submitAnalysisForm(analysisFormRef)"
      >
        <el-form-item label="Дата рентгена" prop="date">
          <el-date-picker
              v-model="analysisForm.date"
              type="date"
              placeholder="Выберите дату"
              style="width: 100%"
              format="DD.MM.YYYY"
              value-format="YYYY-MM-DD"
              :disabled-date="disabledFutureDates"
          />
        </el-form-item>


        <el-form-item label="Загрузите изображение рентгена" prop="xrayImageFile">
          <el-upload
              ref="uploadRef"
              class="upload-demo"
              drag
              action="#"
              :auto-upload="false"
              :on-exceed="handleExceed"
              :on-change="handleFileChange"
              :on-remove="handleFileRemove"
              :limit="1"
              accept=".jpg,.jpeg,.png"
              list-type="picture"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              Перетащите файл сюда или <em>нажмите для выбора</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">
                Файлы JPG/PNG, не более 10MB (проверка на клиенте)
              </div>
            </template>
          </el-upload>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="addAnalysisDialogVisible = false" :disabled="isSubmittingAnalysis">Отмена</el-button>
          <el-button type="primary" @click="submitAnalysisForm(analysisFormRef)" :loading="isSubmittingAnalysis">
            Отправить
          </el-button>
        </span>
      </template>
    </el-dialog>

    <!-- Модалка для просмотра информации об анализе -->
    <el-dialog
        v-model="analysisDetailDialogVisible"
        title="Детали анализа"
        width="700px"
        @closed="selectedAnalysis = null"
    >
      <div v-if="selectedAnalysis" class="analysis-detail-content">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="Дата рентгена">{{ formatDate(selectedAnalysis.date) }}</el-descriptions-item>
          <el-descriptions-item label="Предсказанный возраст">
            {{ selectedAnalysis.predictedAge !== null && selectedAnalysis.predictedAge !== undefined ? selectedAnalysis.predictedAge.toFixed(1) + ' лет' : 'Обработка...' }}
          </el-descriptions-item>
        </el-descriptions>

            <p><strong>Рентгеновский снимок:</strong></p>
            <img
                :src="getFullImageUrl(selectedAnalysis.xrayImageURL)"
                alt="Рентген"
                class="xray-image-preview"
                v-if="selectedAnalysis.xrayImageURL"
                @error="onImageError"
            />
            <el-empty description="Изображение отсутствует или не загрузилось" v-else />
            <p><strong>Примечания доктора:</strong></p>
            <el-form :model="notesForm" @submit.prevent="saveNotes">
              <el-form-item>
                <el-input
                    v-model="notesForm.doctorNotes"
                    type="textarea"
                    rows="6"
                    placeholder="Введите примечания..."
                />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="saveNotes" :loading="isSavingNotes">Сохранить примечания</el-button>
              </el-form-item>
            </el-form>
      </div>
      <el-empty v-else description="Данные анализа не загружены."/>
    </el-dialog>

  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, nextTick } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import type { FormInstance, FormRules, UploadFile, UploadInstance, UploadRawFile } from 'element-plus';
import { ElMessage } from 'element-plus';
import { Back, Plus, UploadFilled } from '@element-plus/icons-vue';
import apiClient, { getFullImageUrl } from '@/services/api'; // Убедитесь, что путь правильный

// Типы, которые приходят с бэкенда
interface AnalysisBackend {
  id: number;
  date: string;
  predictedAge: number | null; // Может быть null, пока обрабатывается
  xrayImageURL: string;
  doctorNotes?: string;
}


interface PatientBackend {
  id: number;
  lastName: string;
  firstName: string;
  middleName?: string;
  birthDate: string;
  gender: 'male' | 'female';
  policyNumber: string;
  analyses: AnalysisBackend[];
}

const route = useRoute();
const router = useRouter();

const loadingPatient = ref(true);
const patient = ref<PatientBackend | null>(null);
const patientId = computed(() => {
  const idParam = route.params.id;
  return Array.isArray(idParam) ? Number(idParam[0]) : Number(idParam);
});


const fetchPatientDetails = async () => {
  if (isNaN(patientId.value)) {
    ElMessage.error('Некорректный ID пациента.');
    loadingPatient.value = false;
    router.push({ name: 'home' });
    return;
  }
  loadingPatient.value = true;
  try {
    const response = await apiClient.get<PatientBackend>(`/patients/${patientId.value}`);
    patient.value = response.data;
  } catch (error: any) {
    ElMessage.error(`Не удалось загрузить данные пациента: ${error.message || 'Ошибка сервера'}`);
    patient.value = null; // Если ошибка, сбрасываем пациента
    router.push({ name: 'home' }); // Можно перенаправить на список
  } finally {
    loadingPatient.value = false;
  }
};

const sortedAnalyses = computed(() => {
  if (!patient.value || !patient.value.analyses) return [];
  return [...patient.value.analyses].sort((a, b) =>
      new Date(b.date).getTime() - new Date(a.date).getTime()
  );
});

// Для модалки добавления анализа
const analysisFormRef = ref<FormInstance>();
const uploadRef = ref<UploadInstance>();
const addAnalysisDialogVisible = ref(false);
const isSubmittingAnalysis = ref(false);

const analysisForm = ref<{
  date: string;
  xrayImageFile: UploadRawFile | null;
}>({
  date: '',
  xrayImageFile: null
});

const analysisRules: FormRules = {
  date: [{ required: true, message: 'Пожалуйста, выберите дату', trigger: 'change' }],
  xrayImageFile: [{
    required: true,
    message: 'Пожалуйста, загрузите изображение рентгена',
    // Валидатор сработает при изменении xrayImageFile
    validator: (rule, value, callback) => {
      if (!analysisForm.value.xrayImageFile) {
        callback(new Error('Пожалуйста, загрузите изображение рентгена'));
      } else {
        callback();
      }
    },
    trigger: 'change'
  }]
};

// Для модалки деталей анализа
const analysisDetailDialogVisible = ref(false);
const selectedAnalysis = ref<AnalysisBackend | null>(null);
const notesForm = ref({ doctorNotes: '' });
const isSavingNotes = ref(false);

const formatDate = (dateString: string): string => {
  if (!dateString) return '-';
  try {
    const [year, month, day] = dateString.split('-');
    if (year && month && day) {
      return `${day}.${month}.${year}`;
    }
    const date = new Date(dateString);
    if (!isNaN(date.getTime())) {
      return date.toLocaleDateString('ru-RU');
    }
    return dateString;
  } catch (e) {
    return dateString;
  }
};

const disabledFutureDates = (time: Date) => {
  return time.getTime() > Date.now();
};

const goBack = () => {
  router.push({ name: 'home' });
};

const showAddAnalysisDialog = () => {
  analysisForm.value.date = new Date().toISOString().split('T')[0];
  analysisForm.value.xrayImageFile = null;
  addAnalysisDialogVisible.value = true;
  // Сброс валидации формы при открытии
  nextTick(() => {
    analysisFormRef.value?.clearValidate();
    uploadRef.value?.clearFiles();
  });
};

const resetAnalysisFormDialog = () => {
  if (analysisFormRef.value) {
    analysisFormRef.value.resetFields(); // Сброс значений полей формы
  }
  analysisForm.value.xrayImageFile = null; // Явный сброс файла
  if (uploadRef.value) {
    uploadRef.value.clearFiles(); // Очистка списка файлов в el-upload
  }
};

function handleExceed(files: File[]){
  uploadRef.value!.clearFiles()
  const file = files[0] as UploadRawFile
  uploadRef.value!.clearFiles()
  uploadRef.value!.handleStart(file)
  uploadRef.value!.submit()
}


const handleFileChange = (uploadFile: UploadFile, uploadFiles: UploadFile[]) => {
  const file = uploadFile.raw;
  if (!file) {
    analysisForm.value.xrayImageFile = null;
    return;
  }

  const isImage = file.type.startsWith('image/jpeg') || file.type.startsWith('image/png');
  const isLt10M = file.size / 1024 / 1024 < 10;

  if (!isImage) {
    ElMessage.error('Можно загружать только изображения JPG/PNG!');
    uploadFiles.splice(0, uploadFiles.length); // Очищаем список файлов
    analysisForm.value.xrayImageFile = null;
    return false;
  }
  if (!isLt10M) {
    ElMessage.error('Размер изображения не должен превышать 10MB!');
    uploadFiles.splice(0, uploadFiles.length);
    analysisForm.value.xrayImageFile = null;
    return false;
  }
  analysisForm.value.xrayImageFile = file;
  // Триггер валидации для поля файла
  analysisFormRef.value?.validateField('xrayImageFile').catch(() => {});

  uploadRef.value!.clearFiles();
  uploadRef.value!.handleStart(file);

  return true;
};

const handleFileRemove = () => {
  analysisForm.value.xrayImageFile = null;
  analysisFormRef.value?.validateField('xrayImageFile').catch(() => {});
};

const submitAnalysisForm = async (formEl: FormInstance | undefined) => {
  if (!formEl || !patient.value) return;
  isSubmittingAnalysis.value = true;

  await formEl.validate(async (valid) => {
    if (valid && analysisForm.value.xrayImageFile) {
      const formData = new FormData();
      formData.append('date', analysisForm.value.date);
      formData.append('xrayImage', analysisForm.value.xrayImageFile as Blob);

      try {
        const response = await apiClient.post<AnalysisBackend>(
            `/patients/${patient.value!.id}/analyses`,
            formData // Axios сам установит Content-Type: multipart/form-data
        );
        const newAnalysis = response.data;
        patient.value!.analyses.push(newAnalysis); // Добавляем новый анализ в список

        ElMessage.success('Анализ успешно добавлен и отправлен на обработку.');
        addAnalysisDialogVisible.value = false;
      } catch (error: any) {
        ElMessage.error(`Ошибка при добавлении анализа: ${error.message || 'Ошибка сервера'}`);
      } finally {
        isSubmittingAnalysis.value = false;
      }
    } else {
      if (!analysisForm.value.xrayImageFile) {
        ElMessage.warning('Пожалуйста, загрузите изображение рентгена.');
      } else {
        ElMessage.warning('Пожалуйста, заполните все обязательные поля.');
      }
      isSubmittingAnalysis.value = false;
    }
  });
};

const showAnalysisDetail = (analysis: AnalysisBackend) => {
  selectedAnalysis.value = { ...analysis }; // Создаем копию, чтобы избежать прямой мутации
  notesForm.value.doctorNotes = analysis.doctorNotes || '';
  analysisDetailDialogVisible.value = true;
};

const saveNotes = async () => {
  if (!selectedAnalysis.value || !patient.value) return;
  isSavingNotes.value = true;
  try {
    const payload = { doctorNotes: notesForm.value.doctorNotes };
    const response = await apiClient.put<AnalysisBackend>(
        `/patients/${patient.value.id}/analyses/${selectedAnalysis.value.id}`,
        payload
    );
    const updatedAnalysisFromServer = response.data;

    const analysisIndex = patient.value.analyses.findIndex(a => a.id === updatedAnalysisFromServer.id);
    if (analysisIndex !== -1) {
      patient.value.analyses[analysisIndex] = updatedAnalysisFromServer;
    }
    if(selectedAnalysis.value && selectedAnalysis.value.id === updatedAnalysisFromServer.id){
      selectedAnalysis.value.doctorNotes = updatedAnalysisFromServer.doctorNotes;
      selectedAnalysis.value.predictedAge = updatedAnalysisFromServer.predictedAge; // Обновляем и возраст, если он мог измениться
    }

    ElMessage.success('Примечания сохранены.');
  } catch (error: any) {
    ElMessage.error(`Ошибка при сохранении примечаний: ${error.message || 'Ошибка сервера'}`);
  } finally {
    isSavingNotes.value = false;
  }
};


const onImageError = (event: Event) => {
  const imgElement = event.target as HTMLImageElement;
  ElMessage.warning('Не удалось загрузить изображение рентгена.');
  if (selectedAnalysis.value) {
    // Можно установить флаг, что изображение не загрузилось, или заменить URL на placeholder
    // selectedAnalysis.value.xrayImageURL = '/path/to/placeholder.png';
  }
};

onMounted(() => {
  fetchPatientDetails();
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

.upload-demo {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.analysis-detail {
  padding: 10px;
}

.xray-image {
  text-align: center;
}
.xray-image-preview {
  width: 100%;
}
</style>

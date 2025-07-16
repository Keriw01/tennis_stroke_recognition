function oAnnotations = GetAnnotations(p_oElan, p_sTierName)

%Zwraca wszystkie adnotacje z warstwy o nazwie p_sTierName obiektu 
%p_oElan.
%
%parametry:
%
%p_oElan - obiekt, na kt�rego danych ma operowa� funkcja. Parametr ten
%musi by� obiektem struktury oElan. W przeciwnym wypadku funkcja 
%zostaje przerwana z b��dem.
%
%p_sTierName - nazwa warstwy, z kt�rej maj� zosta� pobrane adnotacje.
%Warstwa o danej nazwie musi istnie�. Parametr ten musi by� typu 
%string. W przeciwnym wypadku funkcja zostaje przerwana z b��dem.
%
%warto�� zwracana:
%
%Tablica obiekt�w struktury oAnnotation.

	if isstruct(p_oElan) == 0
		oAnnotations = [];
		disp('Error: p_oElan is not a structure');
		return;
	end
	
	if ~isfield(p_oElan, 'oTiers') || ~isfield(p_oElan.oTiers, 'oAnnotations') || ~isfield(p_oElan.oTiers, 'sName')
		oAnnotations = [];
		disp('Error: p_oElan is a wrong structure');
		return;
	end
	
	if ischar(p_sTierName) == 0 
		oAnnotations = [];
		disp('Error: p_sTierName is not a string.');
		return;
	end
	
	bFound = false;
	for i = 1 : numel(p_oElan.oTiers)
		sTemp = p_oElan.oTiers(i).sName;
		if strcmp(p_sTierName, sTemp)
			oAnnotations = p_oElan.oTiers(i).oAnnotations;
			bFound = true;
		end
	end
	
	if ~bFound
		oAnnotations = [];
		disp('Error: Tier of such name does not exist.');
	end
	
end